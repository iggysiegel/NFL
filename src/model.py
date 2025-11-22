"""A state-space model for tracking and predicting NFL team strength."""

import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor import scan

from src.paths import MODEL_DIR

# Suppress PyTensor warnings
warnings.filterwarnings(
    "ignore",
    message="batched_dot is deprecated",
    category=FutureWarning,
    module="pytensor.tensor.blas",
)
warnings.filterwarnings(
    "ignore",
    message="overflow encountered in dot",
    category=RuntimeWarning,
)


class StateSpaceModel:
    """A state-space model for NFL team strength evolution.

    This model incorporates:
        - Time-varying team abilities (state-space evolution).
        - Contextual home field advantage.
        - Hierarchical QB effects.

    Attributes:
        num_teams: Number of NFL teams.
        num_ctx: Number of contextual features.
        num_qbs: Number of unique QBs in the data.
        team_to_idx: Mapping from team names to integer indices.
        qb_to_idx: Mapping from QB names to integer indices.
        trace: PyMC inference data containing posterior samples.
        model: PyMC model object.

    Example:
        >>> model = StateSpaceModel()
        >>> model.fit(historical_games)
        >>> predictions = model.predict(upcoming_games)
    """

    def __init__(self):
        """Initialize the state-space model."""
        self.num_teams = 32
        self.num_ctx = 6
        self.num_qbs = None
        self.team_to_idx = None
        self.qb_to_idx = None
        self.trace = None
        self.model = None

    def prepare_data(self, data: pd.DataFrame) -> tuple:
        """Prepare training data for model fitting.

        Transforms raw game data into the format required by the
        state-space model. Creates separate design matrices for teams
        and QBs, and tracks season transitions.

        Args:
            data: DataFrame containing game results with columns:
                - season: Year of the game.
                - week: Week number within the season.
                - home_team: Name of home team.
                - away_team: Name of away team.
                - is_neutral: Binary flag for neutral site games.
                - is_divisional: Binary flag for divisional games.
                - is_playoff: Binary flag for playoff games.
                - home_grass_to_turf: Home cross-surface flag.
                - home_turf_to_grass: Home cross-surface flag.
                - away_grass_to_turf: Away cross-surface flag.
                - away_turf_to_grass: Away cross-surface flag.
                - rest_advantage: Rest days advantage.
                - home_qb_id: ID of home team's QB.
                - away_qb_id: ID of away team's QB.
                - result: Point differential (home_score - away_score).

        Returns:
            Tuple containing:
                - team_mats: List of team design matrices.
                - qb_mats: List of QB design matrices.
                - y_obs: List of point differentials.
                - is_new_season: Array indicating season transitions.
                - num_steps: Total number of time steps (weeks) in data.
        """
        train_data = data.copy()

        # Create team index mapping
        teams = sorted(
            pd.concat([train_data["home_team"], train_data["away_team"]]).unique()
        )
        assert self.num_teams == len(teams), "Invalid data."
        self.team_to_idx = {team: i for i, team in enumerate(teams)}
        train_data["home_idx"] = train_data["home_team"].map(self.team_to_idx)
        train_data["away_idx"] = train_data["away_team"].map(self.team_to_idx)

        # Create QB index mapping (pool single-game QBs at index 0)
        qb_counts = pd.concat(
            [train_data["home_qb_id"], train_data["away_qb_id"]]
        ).value_counts()
        single_game_qbs = set(qb_counts[qb_counts == 1].index)
        multiple_game_qbs = sorted([qb for qb, c in qb_counts.items() if c > 1])

        self.qb_to_idx = {qb: 0 for qb in single_game_qbs}
        self.qb_to_idx.update({qb: i + 1 for i, qb in enumerate(multiple_game_qbs)})
        self.num_qbs = len(multiple_game_qbs) + 1

        train_data["home_qb_idx"] = train_data["home_qb_id"].map(self.qb_to_idx)
        train_data["away_qb_idx"] = train_data["away_qb_id"].map(self.qb_to_idx)

        # Create season indices
        seasons = sorted(train_data["season"].unique())
        season_to_idx = {season: i for i, season in enumerate(seasons)}
        train_data["season_idx"] = train_data["season"].map(season_to_idx)

        # Create week indices
        weeks = sorted(train_data["week"].unique())
        week_to_idx = {week: i for i, week in enumerate(weeks)}
        train_data["week_idx"] = train_data["week"].map(week_to_idx)

        # Create global step indices across all seasons
        all_steps = (
            train_data[["season_idx", "week_idx"]]
            .drop_duplicates()
            .sort_values(["season_idx", "week_idx"])
            .reset_index(drop=True)
        )
        all_steps["step_idx"] = np.arange(len(all_steps))
        train_data = pd.merge(train_data, all_steps, on=["season_idx", "week_idx"])

        # Build per-week data
        num_steps = train_data["step_idx"].max()
        week_data = []
        for step in range(num_steps + 1):
            week_data.append(train_data[train_data["step_idx"] == step])

        # Identify season transitions
        is_new_season = (
            np.array([wd["week_idx"].iloc[0] for wd in week_data]) == 0
        ).astype("int32")

        # Create design matrices and observation vectors
        team_mats = []
        qb_mats = []
        y_obs = []

        for i in range(num_steps + 1):
            team_x, qb_x = self.build_design_matrices(week_data[i])
            team_mats.append(team_x)
            qb_mats.append(qb_x)
            y_obs.append(week_data[i]["result"].values)

        return team_mats, qb_mats, y_obs, is_new_season, num_steps

    def build_design_matrices(self, week_data: pd.DataFrame) -> tuple:
        """Build separate design matrices for teams and QBs.

        Args:
            week_data: DataFrame containing games for a single week.

        Returns:
            Tuple of (team_design_matrix, qb_design_matrix):
                - team_design_matrix:
                    shape (n_games, num_teams + num_ctx)
                    num_teams columns: +1 for home, -1 for away
                    num_ctx columns: contextual features
                - qb_design_matrix:
                    shape (n_games, num_qbs)
                    num_qbs columns: +1 for home, -1 for away
        """
        n_games = len(week_data)

        # Team design matrix
        team_design = np.zeros((n_games, self.num_teams + self.num_ctx))

        # QB design matrix
        qb_design = np.zeros((n_games, self.num_qbs))

        for i, row in enumerate(week_data.itertuples()):
            # Teams
            home = int(row.home_idx)
            away = int(row.away_idx)
            ctx_vals = [
                1 - row.is_neutral,
                row.is_divisional,
                row.is_playoff,
                row.away_grass_to_turf - row.home_grass_to_turf,
                row.away_turf_to_grass - row.home_turf_to_grass,
                row.rest_advantage,
            ]
            team_design[i, home] = 1.0
            team_design[i, away] = -1.0
            team_design[i, self.num_teams : self.num_teams + self.num_ctx] = ctx_vals

            # QBs
            home_qb = int(row.home_qb_idx)
            away_qb = int(row.away_qb_idx)
            qb_design[i, home_qb] = 1
            qb_design[i, away_qb] = -1

        return team_design, qb_design

    def build_model(
        self,
        team_mats: list,
        qb_mats: list,
        y_obs: list,
        is_new_season: np.ndarray,
        num_steps: int,
    ) -> pm.Model:
        """Build the full PyMC state-space model.

        Args:
            team_mats: List of team design matrices for each week.
            qb_mats: List of QB design matrices for each week.
            y_obs: List of observed point differentials for each week.
            is_new_season: Binary indicators for season transitions.
            num_steps: Total number of time steps.

        Returns:
            Compiled PyMC model ready for inference.
        """
        with pm.Model() as model:
            # ---------------------------------------------------
            # Team Ability Priors
            # ---------------------------------------------------
            phi = pm.Gamma("phi", alpha=0.5, beta=0.5 * 100)
            omega_s = pm.Gamma("omega_s", alpha=0.5, beta=0.5 / 16)
            omega_w = pm.Gamma("omega_w", alpha=0.5, beta=0.5 / 60)
            beta_s = pm.Normal("beta_s", mu=0.98, sigma=1)
            beta_w = pm.Normal("beta_w", mu=0.995, sigma=1)
            omega_zero = pm.Gamma("omega_0", alpha=0.5, beta=0.5 / 6)

            # ---------------------------------------------------
            # HFA Priors
            # ---------------------------------------------------
            alpha_base = pm.Normal("alpha_base", mu=2, sigma=1)
            alpha_divisional = pm.Normal("alpha_divisional", mu=0, sigma=1)
            alpha_playoff = pm.Normal("alpha_playoff", mu=0, sigma=1)
            alpha_turf = pm.Normal("alpha_turf", mu=0, sigma=1)
            alpha_grass = pm.Normal("alpha_grass", mu=0, sigma=1)
            alpha_rest = pm.Normal("alpha_rest", mu=0, sigma=1)

            # ---------------------------------------------------
            # QB Ability Priors
            # ---------------------------------------------------
            qb_variability = pm.HalfNormal("qb_variability", sigma=1.0)
            qb_raw_effect = pm.Normal(
                "qb_raw_effect", mu=0.0, sigma=1.0, shape=self.num_qbs
            )
            qb_scaled_effect = qb_variability * qb_raw_effect
            qb_ability = pm.Deterministic(
                "qb_ability", qb_scaled_effect - pm.math.mean(qb_scaled_effect)
            )

            # ---------------------------------------------------
            # State Evolution
            # ---------------------------------------------------
            theta_init = pm.Normal(
                "theta_0",
                mu=0,
                sigma=1 / pt.sqrt(omega_zero * phi),
                shape=self.num_teams,
            )

            innovation_noise = pm.Normal(
                "innovation_noise", mu=0, sigma=1, shape=(num_steps, self.num_teams)
            )

            def evolve_theta(
                is_new_season_t,
                innovation_t,
                theta_prev,
                beta_s,
                beta_w,
                omega_s,
                omega_w,
                phi,
            ):
                beta_t = is_new_season_t * beta_s + (1 - is_new_season_t) * beta_w
                omega_t = is_new_season_t * omega_s + (1 - is_new_season_t) * omega_w
                g_theta_prev = theta_prev - pt.mean(theta_prev)
                theta_new = beta_t * g_theta_prev + innovation_t / pt.sqrt(
                    omega_t * phi
                )
                return theta_new

            is_new_season_pt = pt.as_tensor(is_new_season[1:])
            theta_sequence, _ = scan(
                fn=evolve_theta,
                sequences=[is_new_season_pt, innovation_noise],
                outputs_info=[theta_init],
                non_sequences=[beta_s, beta_w, omega_s, omega_w, phi],
            )

            all_theta = pt.concatenate([theta_init[None, :], theta_sequence], axis=0)
            pm.Deterministic("theta", all_theta)

            # ---------------------------------------------------
            # Likelihood
            # ---------------------------------------------------
            # Pad both design matrices to same length
            max_games = max(x.shape[0] for x in team_mats)

            team_padded = []
            qb_padded = []
            y_padded = []
            mask = []

            for team_x, qb_x, y in zip(team_mats, qb_mats, y_obs):
                n_games = team_x.shape[0]
                if n_games < max_games:
                    pad_size = max_games - n_games
                    team_pad = np.vstack(
                        [team_x, np.zeros((pad_size, self.num_teams + self.num_ctx))]
                    )
                    qb_pad = np.vstack([qb_x, np.zeros((pad_size, self.num_qbs))])
                    y_pad = np.concatenate([y, np.zeros(pad_size)])
                    m = np.concatenate([np.ones(n_games), np.zeros(pad_size)])
                else:
                    team_pad = team_x
                    qb_pad = qb_x
                    y_pad = y
                    m = np.ones(n_games)

                team_padded.append(team_pad)
                qb_padded.append(qb_pad)
                y_padded.append(y_pad)
                mask.append(m)

            # Convert to arrays
            team_all = np.array(team_padded, dtype=np.float32)
            qb_all = np.array(qb_padded, dtype=np.float32)
            y_all = np.array(y_padded, dtype=np.float32)
            mask_all = np.array(mask, dtype=np.float32)

            # Combine team abilities with contextual home field advantage
            alphas = [
                alpha_base,
                alpha_divisional,
                alpha_playoff,
                alpha_turf,
                alpha_grass,
                alpha_rest,
            ]
            alpha_columns = [pt.tile(a, (num_steps + 1, 1)) for a in alphas]
            theta_expanded = pt.concatenate([all_theta] + alpha_columns, axis=1)

            # Compute team-based predictions: X_team @ [theta; alpha]
            mu_team = pt.batched_dot(team_all, theta_expanded[:, :, None])[:, :, 0]

            # Compute QB-based predictions: X_qb @ qb_ability
            qb_ability_expanded = pt.tile(qb_ability, (num_steps + 1, 1))
            mu_qb = pt.batched_dot(qb_all, qb_ability_expanded[:, :, None])[:, :, 0]

            # Total prediction = team strength + HFA + QB difference
            mu_all = mu_team + mu_qb

            sigma_step = 1 / pm.math.sqrt(phi)

            # Likelihood (only for non-padded observations)
            pm.Normal(
                "y_obs",
                mu=mu_all[mask_all == 1],
                sigma=sigma_step,
                observed=y_all[mask_all == 1],
            )

        return model

    def fit(
        self,
        data: pd.DataFrame,
        draws: int = 1000,
        tune: int = 7000,
        target_accept: float = 0.95,
        chains: int = 4,
        cores: int = 4,
        random_seed: int = 42,
    ) -> None:
        """Fit the state-space model to historical game data.

        Args:
            data: DataFrame of historical games (see prepare_data).
            draws: Number of posterior samples per chain.
            tune: Number of tuning steps.
            target_accept: Target acceptance probability for NUTS.
            chains: Number of MCMC chains.
            cores: Number of CPU cores to use.
            random_seed: Random seed for reproducibility.
        """
        # Prepare data
        team_mats, qb_mats, y_obs, is_new_season, num_steps = self.prepare_data(data)

        # Build model
        self.model = self.build_model(
            team_mats, qb_mats, y_obs, is_new_season, num_steps
        )

        # Sample from posterior
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                target_accept=target_accept,
                chains=chains,
                cores=cores,
                random_seed=random_seed,
                return_inferencedata=True,
            )

    def _get_posterior(self, var: str) -> np.ndarray:
        """Get the posterior for a given variable from the model trace.

        Args:
            var: Name of the variable to extract.

        Returns:
            Posterior samples as a NumPy array.
        """
        if isinstance(self.trace, dict):
            return self.trace[var]
        else:
            return self.trace.posterior[var].values.astype(np.float32)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions with full Bayesian uncertainty.

        Returns DataFrame with:
            - Game identifiers
            - Win probability
            - Full point prediction distribution
            - Team strengths
            - QB effects
            - HFA components

        Args:
            data: Test data with columns matching training data format.

        Returns:
            A DataFrame with predictions results.
        """
        if self.trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Extract posterior samples
        theta = self._get_posterior("theta")
        mean_theta = np.mean(theta[:, :, -1, :], axis=-1, keepdims=True)
        beta_s = self._get_posterior("beta_s")
        beta_w = self._get_posterior("beta_w")
        alpha_base = self._get_posterior("alpha_base")
        alpha_divisional = self._get_posterior("alpha_divisional")
        alpha_playoff = self._get_posterior("alpha_playoff")
        alpha_turf = self._get_posterior("alpha_turf")
        alpha_grass = self._get_posterior("alpha_grass")
        alpha_rest = self._get_posterior("alpha_rest")
        qb_ability = self._get_posterior("qb_ability")

        # Iterate through test data
        results = []
        for row in data.itertuples():

            # Get team indices
            home_team_idx = self.team_to_idx[row.home_team]
            away_team_idx = self.team_to_idx[row.away_team]

            # Select evolution parameter
            beta = beta_s if row.week == 1 else beta_w
            beta = beta[:, :, None]

            # Compute centered team abilities
            home_team_strength = beta * (
                theta[:, :, -1, home_team_idx : home_team_idx + 1] - mean_theta
            )
            away_team_strength = beta * (
                theta[:, :, -1, away_team_idx : away_team_idx + 1] - mean_theta
            )

            # Get QB indices
            home_qb_idx = self.qb_to_idx.get(row.home_qb_id, 0)
            away_qb_idx = self.qb_to_idx.get(row.away_qb_id, 0)

            # Add QB effects
            home_qb_effect = qb_ability[:, :, home_qb_idx : home_qb_idx + 1]
            away_qb_effect = qb_ability[:, :, away_qb_idx : away_qb_idx + 1]

            # Add home field advantage
            hfa_base = (1 - row.is_neutral) * alpha_base[:, :, None]
            hfa_divisional = row.is_divisional * alpha_divisional[:, :, None]
            hfa_playoff = row.is_playoff * alpha_playoff[:, :, None]
            hfa_turf = (row.away_grass_to_turf - row.home_grass_to_turf) * alpha_turf[
                :, :, None
            ]
            hfa_grass = (row.away_turf_to_grass - row.home_turf_to_grass) * alpha_grass[
                :, :, None
            ]
            hfa_rest = row.rest_advantage * alpha_rest[:, :, None]
            hfa = (
                hfa_base
                + hfa_divisional
                + hfa_playoff
                + hfa_turf
                + hfa_grass
                + hfa_rest
            )

            # Predicted point differential
            prediction = (
                home_team_strength
                + home_qb_effect
                - away_team_strength
                - away_qb_effect
                + hfa
            ).reshape(-1)
            prediction_percentiles = np.percentile(prediction, list(range(1, 100)))

            # Append results
            results.append(
                [
                    # Game identifiers
                    row.season,
                    row.week,
                    row.home_team,
                    row.away_team,
                    # Win probability
                    float((prediction > 0).mean()),
                    float((prediction < 0).mean()),
                    # Prediction distribution
                    float(prediction.mean()),
                    float(prediction.std()),
                    *prediction_percentiles,
                    # Team strengths
                    float(home_team_strength.mean()),
                    float(home_team_strength.std()),
                    float(away_team_strength.mean()),
                    float(away_team_strength.std()),
                    # QB effects
                    float(home_qb_effect.mean()),
                    float(home_qb_effect.std()),
                    float(away_qb_effect.mean()),
                    float(away_qb_effect.std()),
                    # HFA components
                    float(hfa.mean()),
                    float(hfa.std()),
                    float(hfa_base.mean()),
                    float(hfa_base.std()),
                    float(hfa_divisional.mean()),
                    float(hfa_divisional.std()),
                    float(hfa_playoff.mean()),
                    float(hfa_playoff.std()),
                    float(hfa_turf.mean()),
                    float(hfa_turf.std()),
                    float(hfa_grass.mean()),
                    float(hfa_grass.std()),
                    float(hfa_rest.mean()),
                    float(hfa_rest.std()),
                ]
            )

        # Return DataFrame
        columns = (
            # Game identifiers
            ["season", "week", "home_team", "away_team"]
            # Win probability
            + ["home_win_prob", "away_win_prob"]
            # Prediction distribution
            + ["prediction_mean", "prediction_std"]
            + [f"prediction_ci_{p:02d}" for p in list(range(1, 100))]
            # Team strengths
            + [
                "home_team_strength_mean",
                "home_team_strength_std",
                "away_team_strength_mean",
                "away_team_strength_std",
            ]
            # QB effects
            + [
                "home_qb_effect_mean",
                "home_qb_effect_std",
                "away_qb_effect_mean",
                "away_qb_effect_std",
            ]
            # HFA components
            + [
                "hfa_mean",
                "hfa_std",
                "hfa_base_mean",
                "hfa_base_std",
                "hfa_divisional_mean",
                "hfa_divisional_std",
                "hfa_playoff_mean",
                "hfa_playoff_std",
                "hfa_turf_mean",
                "hfa_turf_std",
                "hfa_grass_mean",
                "hfa_grass_std",
                "hfa_rest_mean",
                "hfa_rest_std",
            ]
        )

        return pd.DataFrame(results, columns=columns)

    def save_model(self) -> None:
        """Save the fitted model to disk and delete files in the model directory before
        saving."""
        if self.trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        for file in MODEL_DIR.glob("*"):
            if file.is_file():
                file.unlink()

        arrays = {
            "theta": self._get_posterior("theta"),
            "beta_s": self._get_posterior("beta_s"),
            "beta_w": self._get_posterior("beta_w"),
            "alpha_base": self._get_posterior("alpha_base"),
            "alpha_divisional": self._get_posterior("alpha_divisional"),
            "alpha_playoff": self._get_posterior("alpha_playoff"),
            "alpha_turf": self._get_posterior("alpha_turf"),
            "alpha_grass": self._get_posterior("alpha_grass"),
            "alpha_rest": self._get_posterior("alpha_rest"),
            "qb_ability": self._get_posterior("qb_ability"),
        }
        arrays["team_to_idx"] = np.array([self.team_to_idx], dtype=object)
        arrays["qb_to_idx"] = np.array([self.qb_to_idx], dtype=object)

        np.savez_compressed(MODEL_DIR / "model.npz", **arrays)

    def load_model(self) -> None:
        """Load a previously saved model from disk."""
        model_path = MODEL_DIR / "model.npz"
        if not model_path.exists():
            raise FileNotFoundError("Model file not found.")

        arrays = np.load(model_path, allow_pickle=True)

        self.trace = {
            "theta": arrays["theta"],
            "beta_s": arrays["beta_s"],
            "beta_w": arrays["beta_w"],
            "alpha_base": arrays["alpha_base"],
            "alpha_divisional": arrays["alpha_divisional"],
            "alpha_playoff": arrays["alpha_playoff"],
            "alpha_turf": arrays["alpha_turf"],
            "alpha_grass": arrays["alpha_grass"],
            "alpha_rest": arrays["alpha_rest"],
            "qb_ability": arrays["qb_ability"],
        }
        self.team_to_idx = arrays["team_to_idx"].item()
        self.qb_to_idx = arrays["qb_to_idx"].item()

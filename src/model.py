"""A state-space model for tracking and predicting NFL team strength."""

import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor import scan

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

    This model treats team abilities as latent states that evolve over
    time according to a mean-reverting process. QB effects are modeled
    hierarchically to account for individual QB talent.

    Attributes:
        num_teams: Number of NFL teams.
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
                - home_qb_id: ID of home team's QB.
                - away_qb_id: ID of away team's QB.
                - is_neutral: Binary flag for neutral site games.
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
        teams = sorted(train_data["home_team"].unique())
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
        all_steps = train_data[["season_idx", "week_idx"]].drop_duplicates()
        all_steps["step_idx"] = np.arange(len(all_steps))
        train_data = pd.merge(train_data, all_steps, on=["season_idx", "week_idx"])

        # Build per-week data
        num_steps = train_data["step_idx"].max()
        week_data = []
        for step in range(num_steps + 1):
            week_data.append(train_data[train_data["step_idx"] == step])

        # Identify season transitions (week 1 of each season)
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
                - team_design_matrix: shape (n_games, num_teams + 1)
                  Last column encodes home field advantage
                - qb_design_matrix: shape (n_games, num_qbs)
                  +1 for home QB, -1 for away QB
        """
        n_games = len(week_data)

        # Team design matrix (existing logic)
        team_design = np.zeros((n_games, self.num_teams + 1))

        # QB design matrix (new)
        qb_design = np.zeros((n_games, self.num_qbs))

        for i, row in enumerate(week_data.itertuples()):
            # Teams
            home = int(row.home_idx)
            away = int(row.away_idx)
            is_home = int(1 - row.is_neutral)
            team_design[i, home] = 1
            team_design[i, away] = -1
            team_design[i, self.num_teams] = is_home

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
        """Build the full PyMC state-space model with QB effects.

        Constructs a hierarchical Bayesian model with:
        - Time-varying team abilities (latent states)
        - Different evolution dynamics for season vs. week transitions
        - Home field advantage parameter
        - Hierarchical QB ability parameters

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
            alpha = pm.Normal("alpha", mu=2, sigma=1)

            # ---------------------------------------------------
            # QB Ability Priors
            # ---------------------------------------------------
            tau_qb = pm.HalfNormal("tau_qb", sigma=2.0)

            qb_delta_raw = pm.Normal(
                "qb_delta_raw", mu=0.0, sigma=1.0, shape=self.num_qbs
            )
            qb_delta = tau_qb * qb_delta_raw

            qb_ability = pm.Deterministic(
                "qb_ability", qb_delta - pm.math.mean(qb_delta)
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
                    team_pad = np.vstack(
                        [team_x, np.zeros((max_games - n_games, self.num_teams + 1))]
                    )
                    qb_pad = np.vstack(
                        [qb_x, np.zeros((max_games - n_games, self.num_qbs))]
                    )
                    y_pad = np.concatenate([y, np.zeros(max_games - n_games)])
                    m = np.concatenate(
                        [np.ones(n_games), np.zeros(max_games - n_games)]
                    )
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

            # Combine team abilities with home field advantage
            alpha_column = pt.tile(alpha, (num_steps + 1, 1))
            theta_expanded = pt.concatenate([all_theta, alpha_column], axis=1)

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

    def fit(self, data: pd.DataFrame) -> None:
        """Fit the state-space model to historical game data.

        Args:
            data: DataFrame of historical games (see prepare_data).
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
                draws=1000,
                tune=9000,
                random_seed=42,
                chains=4,
                cores=4,
                target_accept=0.95,
                return_inferencedata=True,
            )

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions with full Bayesian uncertainty.

        Produces win probabilities, expected score differentials, and
        percentile-based confidence intervals using the full posterior
        distribution. Incorporates QB effects for each matchup.

        Args:
            data: Test data with columns matching training data format.

        Returns:
            A DataFrame with prediction results including columns for
            means, standard deviations, and percentiles (1–99).
        """
        if self.trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Extract posterior samples
        theta = self.trace.posterior["theta"].values
        alpha = self.trace.posterior["alpha"].values
        beta_s = self.trace.posterior["beta_s"].values
        beta_w = self.trace.posterior["beta_w"].values
        qb_ability = self.trace.posterior["qb_ability"].values

        # Compute mean ability for centering
        mean_theta = np.mean(theta[:, :, -1, :], axis=-1, keepdims=True)

        percentiles = list(range(1, 100))
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
            hfa = (
                np.zeros_like(alpha)[:, :, None]
                if row.is_neutral
                else alpha[:, :, None]
            )

            # Predicted point differential (team + QB + HFA)
            prediction = (
                home_team_strength
                + home_qb_effect
                - away_team_strength
                - away_qb_effect
                + hfa
            )

            # Compute summary statistics and percentiles
            pred_percentiles = np.percentile(prediction, percentiles)

            results.append(
                [
                    # Game identifiers
                    row.season,
                    row.week,
                    row.home_team,
                    row.away_team,
                    # Win probabilities
                    float((prediction > 0).mean()),
                    float((prediction < 0).mean()),
                    # Point spread prediction
                    float(prediction.mean()),
                    float(prediction.std()),
                    *pred_percentiles,
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
                    # HFA
                    float(hfa.mean()),
                    float(hfa.std()),
                ]
            )

        # Build column names
        columns = (
            # Game identifiers
            ["season", "week", "home_team", "away_team"]
            # Win probabilities
            + ["home_win_prob", "away_win_prob"]
            # Point spread prediction
            + ["prediction_mean", "prediction_std"]
            + [f"prediction_ci_{p:02d}" for p in percentiles]
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
            # HFA
            + ["hfa_mean", "hfa_std"]
        )

        return pd.DataFrame(results, columns=columns)

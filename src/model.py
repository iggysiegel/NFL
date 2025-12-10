"""A state-space model for predicting NFL game outcomes."""

import json
import warnings
from pathlib import Path

import arviz as az
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
    """A state-space model for predicting NFL game outcomes.

    This model incorporates:
        - Time-varying team abilities.
        - Time-varying QB abilities.
        - Home field advantage.

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

    def prepare_data(self, data: pd.DataFrame) -> dict:
        """Prepare training data for model fitting.

        Transforms raw game data into the format required by the
        state-space model. Tracks time steps and season transitions,
        and creates efficient QB tracking structures and separate
        design matrices for teams and QBs.

        Args:
            data: DataFrame containing game results with columns:
                - season: Year of the game.
                - week: Week number within the season.
                - home_team: Name of home team.
                - away_team: Name of away team.
                - is_neutral: Binary flag for neutral site games.
                - home_qb_id: ID of home team's QB.
                - home_qb_experience: Home team QB career games played.
                - away_qb_id: ID of away team's QB.
                - away_qb_experience: Away team QB career games played.
                - result: Point differential (home_score - away_score).

        Returns:
            A dict with keys:
                - num_steps: Total number of time steps (weeks) in data.
                - is_new_season: Array indicating season transitions.
                - qb_activity: Binary matrix when each QB played.
                - qb_debut: Array indicating first appearance of QBs.
                - qb_experience_at_debut: Array of QB experience.
                - team_mats: List of team design matrices.
                - qb_mats: List of QB design matrices.
                - y_obs: List of point differentials.
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

        # Create QB index mapping
        all_qbs = sorted(
            pd.concat([train_data["home_qb_id"], train_data["away_qb_id"]]).unique()
        )
        self.qb_to_idx = {qb: i for i, qb in enumerate(all_qbs)}
        self.num_qbs = len(all_qbs)
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

        # Create QB tracking, design matrices, and observation vectors
        qb_data = self.build_qb_tracking(train_data, num_steps)

        team_mats = []
        qb_mats = []
        y_obs = []

        for i in range(num_steps + 1):
            team_x, qb_x = self.build_design_matrices(week_data[i])
            team_mats.append(team_x)
            qb_mats.append(qb_x)
            y_obs.append(week_data[i]["result"].values)

        return {
            "num_steps": num_steps,
            "is_new_season": is_new_season,
            "qb_activity": qb_data["qb_activity"],
            "qb_debut": qb_data["qb_debut"],
            "qb_experience_at_debut": qb_data["qb_experience_at_debut"],
            "team_mats": team_mats,
            "qb_mats": qb_mats,
            "y_obs": y_obs,
        }

    def build_qb_tracking(self, data: pd.DataFrame, num_steps: int) -> dict:
        """Build QB activity tracking structures.

        Creates three structures:
            - qb_activity
                Shape (num_steps + 1, num_qbs)
                Binary matrix when each QB played
            - qb_debut
                Shape (num_qbs)
                First time step when each QB appeared
            - qb_experience_at_debut
                Shape (num_qbs)
                Career games at first appearance in data

        Args:
            data: Training dataset with prepare_data() transformations.
            num_steps: Total number of time steps (weeks) in data.

        Returns:
            A dict containing all three structures.
        """
        # QB activity matrix
        qb_activity = np.zeros((num_steps + 1, self.num_qbs), dtype=np.int32)

        for step in range(num_steps + 1):
            step_data = data[data["step_idx"] == step]
            home_qbs = step_data["home_qb_idx"].values
            away_qbs = step_data["away_qb_idx"].values
            qb_activity[step, home_qbs] = 1
            qb_activity[step, away_qbs] = 1

        # QB debut matrix
        qb_debut = np.full(self.num_qbs, -1, dtype=np.int32)

        for qb_idx in range(self.num_qbs):
            appearances = np.where(qb_activity[:, qb_idx] == 1)[0]
            if len(appearances) > 0:
                qb_debut[qb_idx] = appearances[0]

        # QB experience at debut matrix
        qb_experience_at_debut = np.zeros(self.num_qbs, dtype=np.float32)

        for qb_idx in range(self.num_qbs):
            if qb_debut[qb_idx] > -1:
                debut_step = qb_debut[qb_idx]
                debut_games = data[data["step_idx"] == debut_step]
                home_match = debut_games[debut_games["home_qb_idx"] == qb_idx]
                away_match = debut_games[debut_games["away_qb_idx"] == qb_idx]
                if len(home_match) > 0:
                    qb_experience_at_debut[qb_idx] = home_match.iloc[0][
                        "home_qb_experience"
                    ]
                if len(away_match) > 0:
                    qb_experience_at_debut[qb_idx] = away_match.iloc[0][
                        "away_qb_experience"
                    ]

        return {
            "qb_activity": qb_activity,
            "qb_debut": qb_debut,
            "qb_experience_at_debut": qb_experience_at_debut,
        }

    def build_design_matrices(self, week_data: pd.DataFrame) -> tuple:
        """Build separate design matrices for teams and QBs.

        Args:
            week_data: DataFrame containing games for a single week.

        Returns:
            Tuple of (team_design_matrix, qb_design_matrix):
                - team_design_matrix:
                    shape (n_games, num_teams + 1)
                    num_teams columns: +1 for home, -1 for away
                    hfa column: +1 for home, 0 for neutral
                - qb_design_matrix:
                    shape (n_games, num_qbs)
                    num_qbs columns: +1 for home, -1 for away
        """
        n_games = len(week_data)

        # Team design matrix
        team_design = np.zeros((n_games, self.num_teams + 1))

        # QB design matrix
        qb_design = np.zeros((n_games, self.num_qbs))

        for i, row in enumerate(week_data.itertuples()):
            # Teams
            home = int(row.home_idx)
            away = int(row.away_idx)
            team_design[i, home] += 1
            team_design[i, away] -= 1
            team_design[i, self.num_teams] = 1 - row.is_neutral

            # QBs
            home_qb = int(row.home_qb_idx)
            away_qb = int(row.away_qb_idx)
            qb_design[i, home_qb] += 1
            qb_design[i, away_qb] -= 1

        return team_design, qb_design

    def build_model(self, data_dict: dict) -> pm.Model:
        """Build the full PyMC state-space model.

        Args:
            data_dict: Dictionary returned by prepare_data().

        Returns:
            Compiled PyMC model ready for inference.
        """
        num_steps = data_dict["num_steps"]
        is_new_season = data_dict["is_new_season"]
        qb_activity = data_dict["qb_activity"]
        qb_debut = data_dict["qb_debut"]
        qb_experience_at_debut = data_dict["qb_experience_at_debut"]
        team_mats = data_dict["team_mats"]
        qb_mats = data_dict["qb_mats"]
        y_obs = data_dict["y_obs"]

        with pm.Model() as model:
            # Convert numpy arrays to PyTensor tensors
            step_indices_pt = pt.arange(num_steps)
            is_new_season_pt = pt.as_tensor(is_new_season)
            qb_activity_pt = pt.as_tensor(qb_activity)
            qb_debut_pt = pt.as_tensor(qb_debut)
            qb_experience_at_debut_pt = pt.as_tensor(qb_experience_at_debut)

            # ---------------------------------------------------
            # Priors
            # ---------------------------------------------------
            # Overall model precision
            phi = pm.Gamma("phi", alpha=0.5, beta=0.5 * 100)

            # Home field advantage parameter
            alpha_base = pm.Normal("alpha_base", mu=2, sigma=0.5)

            # ---------------------------------------------------
            # Team Evolution
            # ---------------------------------------------------
            # Between-season and between-week evolution precision
            omega_theta_s = pm.Gamma("omega_theta_s", alpha=0.5, beta=0.5 / 16)
            omega_theta_w = pm.Gamma("omega_theta_w", alpha=0.5, beta=0.5 / 60)

            # Between-season and between-week regression parameters
            beta_theta_s = pm.Normal("beta_theta_s", mu=0.98, sigma=1)
            beta_theta_w = pm.Normal("beta_theta_w", mu=0.995, sigma=1)

            # Initial team strength precision
            omega_theta_init = pm.Gamma("omega_theta_init", alpha=0.5, beta=0.5 / 6)

            # Initial team strength parameters
            theta_init = pm.Normal(
                "theta_init",
                mu=0,
                sigma=1 / pt.sqrt(omega_theta_init * phi),
                shape=self.num_teams,
            )

            # Team strength random innovations
            theta_innovation = pm.Normal(
                "theta_innovation", mu=0, sigma=1, shape=(num_steps, self.num_teams)
            )

            # AR(1) update process
            def evolve_theta(
                is_new_season_t,
                theta_innovation_t,
                theta_prev,
                beta_theta_s,
                beta_theta_w,
                omega_theta_s,
                omega_theta_w,
                phi,
            ):
                # Select precision and regression parameters
                omega_theta_t = (
                    is_new_season_t * omega_theta_s
                    + (1 - is_new_season_t) * omega_theta_w
                )
                beta_theta_t = (
                    is_new_season_t * beta_theta_s
                    + (1 - is_new_season_t) * beta_theta_w
                )

                # Center previous strengths
                centering = theta_prev - pt.mean(theta_prev)

                # AR(1) update
                theta_new = beta_theta_t * centering + theta_innovation_t / pt.sqrt(
                    omega_theta_t * phi
                )
                return theta_new

            theta_sequence, _ = scan(
                fn=evolve_theta,
                sequences=[
                    is_new_season_pt[1:],
                    theta_innovation,
                ],
                outputs_info=[theta_init],
                non_sequences=[
                    beta_theta_s,
                    beta_theta_w,
                    omega_theta_s,
                    omega_theta_w,
                    phi,
                ],
            )

            all_theta = pt.concatenate([theta_init[None, :], theta_sequence], axis=0)
            pm.Deterministic("theta", all_theta)

            # ---------------------------------------------------
            # QB Evolution
            # ---------------------------------------------------
            # Between-week evolution precision
            omega_qb_w = pm.Gamma("omega_qb_w", alpha=0.5, beta=0.5 / 100)

            # Between-week regression parameter
            beta_qb_w = pm.Normal("beta_qb_w", mu=0.995, sigma=1)

            # Initial QB strength precision
            omega_qb_init = pm.Gamma("omega_qb_init", alpha=0.5, beta=0.5 / 16)

            # Initial QB strength parameters
            rookie_mean = pm.Normal("rookie_mean", mu=-2.5, sigma=1.0)
            veteran_mean = pm.Normal("veteran_mean", mu=0.0, sigma=1.0)
            experience_factor = pt.minimum(1.0, qb_experience_at_debut_pt / 48.0)
            qb_init_mu = rookie_mean + experience_factor * (veteran_mean - rookie_mean)
            qb_init = pm.Normal(
                "qb_init",
                mu=qb_init_mu,
                sigma=1 / pt.sqrt(omega_qb_init * phi),
                shape=self.num_qbs,
            )

            # QB strength random innovations
            qb_innovation = pm.Normal(
                "qb_innovation", mu=0, sigma=1, shape=(num_steps, self.num_qbs)
            )

            # AR(1) update process
            def evolve_qb(
                step_idx,
                qb_activity_t,
                qb_innovation_t,
                qb_prev,
                omega_qb_w,
                beta_qb_w,
                phi,
                qb_init,
                qb_debut_pt,
            ):
                # Is this each QB's debut week / has each QB debuted this week
                is_debut = pt.eq(step_idx, qb_debut_pt)
                has_debuted = pt.and_(
                    pt.ge(qb_debut_pt, 0), pt.le(qb_debut_pt, step_idx)
                )

                # Center active QBs
                active_sum = pt.sum(qb_prev * has_debuted)
                active_count = pt.sum(has_debuted) + 1e-6
                active_mean = active_sum / active_count
                centering = qb_prev - active_mean

                # Evolution is computed for all QBs that played
                evolved = beta_qb_w * centering + qb_innovation_t / pt.sqrt(
                    omega_qb_w * phi
                )

                # Vectorized new state
                # Level 1: Has the QB debuted yet?
                #   No  - return 0 (placeholder)
                #   Yes - go to Level 2
                #
                # Level 2: Is this their debut week?
                #   Yes - return qb_init (initialize from prior)
                #   No  - go to Level 3
                #
                # Level 3: Did they play this week?
                #   Yes - return evolved (update with innovation)
                #   No  - return qb_prev (freeze at previous value)
                qb_new = pt.where(
                    has_debuted,
                    pt.where(
                        is_debut, qb_init, pt.where(qb_activity_t, evolved, qb_prev)
                    ),
                    pt.zeros_like(qb_prev),
                )
                return qb_new

            qb_sequence, _ = scan(
                fn=evolve_qb,
                sequences=[step_indices_pt, qb_activity_pt[1:], qb_innovation],
                outputs_info=[pt.zeros(self.num_qbs)],
                non_sequences=[omega_qb_w, beta_qb_w, phi, qb_init, qb_debut_pt],
            )

            all_qb = pt.concatenate([pt.zeros((1, self.num_qbs)), qb_sequence], axis=0)
            pm.Deterministic("qb_ability", all_qb)

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
                        [team_x, np.zeros((pad_size, self.num_teams + 1))]
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

            # Compute team-based predictions: X_team @ [theta; alpha]
            alpha_tiled = pt.tile(alpha_base, (num_steps + 1, 1))
            theta_expanded = pt.concatenate([all_theta, alpha_tiled], axis=1)
            mu_team = pt.batched_dot(team_all, theta_expanded[:, :, None])[:, :, 0]

            # Compute QB-based predictions: X_qb @ qb_ability
            mu_qb = pt.batched_dot(qb_all, all_qb[:, :, None])[:, :, 0]

            # Total prediction = team strength + HFA + QB ability
            mu_all = mu_team + mu_qb
            sigma_step = 1 / pt.sqrt(phi)

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
            data: DataFrame of historical games (see prepare_data()).
            draws: Number of posterior samples per chain.
            tune: Number of tuning steps.
            target_accept: Target acceptance probability for NUTS.
            chains: Number of MCMC chains.
            cores: Number of CPU cores to use.
            random_seed: Random seed for reproducibility.
        """
        # Prepare data
        data_dict = self.prepare_data(data)

        # Build model
        self.model = self.build_model(data_dict)

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
        return self.trace.posterior[var].values.astype(np.float32)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions with full Bayesian uncertainty.

        Returns DataFrame with:
            - Game identifiers
            - Win probability
            - Full point prediction distribution
            - Team strengths
            - QB effects
            - HFA

        Args:
            data: Test data with columns matching training data format.

        Returns:
            A DataFrame with predictions results.
        """
        if self.trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Extract posterior samples
        phi = self._get_posterior("phi")
        alpha_base = self._get_posterior("alpha_base")
        beta_theta_s = self._get_posterior("beta_theta_s")
        beta_theta_w = self._get_posterior("beta_theta_w")
        theta = self._get_posterior("theta")
        mean_theta = np.mean(theta[:, :, -1, :], axis=-1, keepdims=True)
        omega_qb_init = self._get_posterior("omega_qb_init")
        rookie_mean = self._get_posterior("rookie_mean")
        qb_ability = self._get_posterior("qb_ability")

        # Iterate through test data
        results = []
        for row in data.itertuples():

            # Get team indices
            home_team_idx = self.team_to_idx[row.home_team]
            away_team_idx = self.team_to_idx[row.away_team]

            # Compute team abilities
            beta_theta = beta_theta_s if row.week == 1 else beta_theta_w
            beta_theta = beta_theta[:, :, None]

            home_team_strength = beta_theta * (
                theta[:, :, -1, home_team_idx : home_team_idx + 1] - mean_theta
            )
            away_team_strength = beta_theta * (
                theta[:, :, -1, away_team_idx : away_team_idx + 1] - mean_theta
            )

            # Compute QB abilities
            if row.home_qb_id in self.qb_to_idx:
                home_qb_idx = self.qb_to_idx[row.home_qb_id]
                home_qb_strength = qb_ability[:, :, -1, home_qb_idx : home_qb_idx + 1]
            else:
                prior_std = (1 / np.sqrt(omega_qb_init * phi)).mean()
                home_qb_strength = np.random.normal(
                    loc=rookie_mean[:, :, None],
                    scale=prior_std,
                    size=(rookie_mean.shape[0], rookie_mean.shape[1], 1),
                )

            if row.away_qb_id in self.qb_to_idx:
                away_qb_idx = self.qb_to_idx[row.away_qb_id]
                away_qb_strength = qb_ability[:, :, -1, away_qb_idx : away_qb_idx + 1]
            else:
                prior_std = (1 / np.sqrt(omega_qb_init * phi)).mean()
                away_qb_strength = np.random.normal(
                    loc=rookie_mean[:, :, None],
                    scale=prior_std,
                    size=(rookie_mean.shape[0], rookie_mean.shape[1], 1),
                )

            # Compute home field advantage
            hfa = (1 - row.is_neutral) * alpha_base[:, :, None]

            # Predicted point differential
            prediction = (
                home_team_strength
                + home_qb_strength
                - away_team_strength
                - away_qb_strength
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
                    float(home_qb_strength.mean()),
                    float(home_qb_strength.std()),
                    float(away_qb_strength.mean()),
                    float(away_qb_strength.std()),
                    # HFA components
                    float(hfa.mean()),
                    float(hfa.std()),
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
                "home_qb_strength_mean",
                "home_qb_strength_std",
                "away_qb_strength_mean",
                "away_qb_strength_std",
            ]
            # HFA components
            + ["hfa_mean", "hfa_std"]
        )

        return pd.DataFrame(results, columns=columns)

    def save(self, model_path: str, overwrite: bool = True) -> None:
        """Save the fitted model to disk.

        Args:
            model_path: File path where the model trace will be saved.
            overwrite: If true, delete existing models in directory.
        """
        if self.trace is None:
            raise ValueError("Model not fitted. Call fit() first.")
        idata = self.trace.copy()

        path = Path(model_path)
        if path.suffix == "":
            path = path.with_suffix(".nc")
        path.parent.mkdir(parents=True, exist_ok=True)

        if overwrite:
            for existing in path.parent.glob("*.nc"):
                existing.unlink()

        idata.attrs["team_to_idx"] = json.dumps(self.team_to_idx)
        idata.attrs["qb_to_idx"] = json.dumps(self.qb_to_idx)

        az.to_netcdf(idata, path)

    def load(self, model_path: str) -> None:
        """Load a previously saved model from disk.

        Args:
            model_path: File path where the model trace will be loaded.
        """
        path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}.")

        self.trace = az.from_netcdf(path)
        self.team_to_idx = json.loads(self.trace.attrs["team_to_idx"])
        self.qb_to_idx = json.loads(self.trace.attrs["qb_to_idx"])

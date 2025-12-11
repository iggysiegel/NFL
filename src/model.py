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
        - Static QB effects.
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
        state-space model. Creates separate design matrices for teams
        and QBs, and tracks season transitions and QB experience.

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
                - qb_experience: Mean career games played by each QB.
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

        # Create QB experience array
        qb_experience = np.full(self.num_qbs, 0.0, dtype=np.float32)
        home_exp = train_data[["home_qb_idx", "home_qb_experience"]].rename(
            columns={"home_qb_idx": "qb_idx", "home_qb_experience": "experience"}
        )
        away_exp = train_data[["away_qb_idx", "away_qb_experience"]].rename(
            columns={"away_qb_idx": "qb_idx", "away_qb_experience": "experience"}
        )
        long_exp = (
            pd.concat([home_exp, away_exp])
            .groupby("qb_idx")["experience"]
            .mean()
            .to_dict()
        )
        for qb_idx in range(self.num_qbs):
            qb_experience[qb_idx] = long_exp.get(qb_idx, 0.0)

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
        num_steps = int(train_data["step_idx"].max() + 1)
        week_data = []
        for step in range(num_steps):
            week_data.append(train_data[train_data["step_idx"] == step])

        # Identify season transitions
        is_new_season = (
            np.array([wd["week_idx"].iloc[0] for wd in week_data]) == 0
        ).astype(np.int32)

        # Create design matrices and observation vectors
        team_mats = []
        qb_mats = []
        y_obs = []

        for i in range(num_steps):
            team_x, qb_x = self.build_design_matrices(week_data[i])
            team_mats.append(team_x)
            qb_mats.append(qb_x)
            y_obs.append(week_data[i]["result"].values.astype(np.float32))

        return {
            "num_steps": num_steps,
            "is_new_season": is_new_season,
            "qb_experience": qb_experience,
            "team_mats": team_mats,
            "qb_mats": qb_mats,
            "y_obs": y_obs,
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
        team_design = np.zeros((n_games, self.num_teams + 1), dtype=np.float32)

        # QB design matrix
        qb_design = np.zeros((n_games, self.num_qbs), dtype=np.float32)

        for i, row in enumerate(week_data.itertuples()):
            # Teams
            home = int(row.home_idx)
            away = int(row.away_idx)
            team_design[i, home] = 1.0
            team_design[i, away] = -1.0
            team_design[i, self.num_teams] = 1.0 - float(row.is_neutral)

            # QBs
            home_qb = int(row.home_qb_idx)
            away_qb = int(row.away_qb_idx)
            qb_design[i, home_qb] = 1.0
            qb_design[i, away_qb] = -1.0

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
        qb_experience = data_dict["qb_experience"]
        team_mats = data_dict["team_mats"]
        qb_mats = data_dict["qb_mats"]
        y_obs = data_dict["y_obs"]

        with pm.Model() as model:
            # Convert numpy arrays to PyTensor tensors
            is_new_season_pt = pt.as_tensor(is_new_season, dtype="int32")
            qb_experience_pt = pt.as_tensor(qb_experience, dtype="float32")

            # ---------------------------------------------------
            # Global Parameters
            # ---------------------------------------------------
            # Overall model precision
            phi = pm.Gamma("phi", alpha=0.5, beta=0.5 * 100)

            # Home field advantage parameter
            alpha_hfa = pm.Normal("alpha_hfa", mu=2.0, sigma=0.5)

            # ---------------------------------------------------
            # Team Strength Dynamics
            # ---------------------------------------------------
            # Between-season and between-week evolution precision
            omega_team_season = pm.Gamma("omega_team_season", alpha=0.5, beta=0.5 / 16)
            omega_team_week = pm.Gamma("omega_team_week", alpha=0.5, beta=0.5 / 60)

            # Between-season and between-week regression parameters
            beta_team_season = pm.Normal("beta_team_season", mu=0.98, sigma=1.0)
            beta_team_week = pm.Normal("beta_team_week", mu=0.995, sigma=1.0)

            # Initial team strength precision
            omega_team_init = pm.Gamma("omega_team_init", alpha=0.5, beta=0.5 / 6)

            # Initial team strength parameters
            team_strength_init = pm.Normal(
                "team_strength_init",
                mu=0.0,
                sigma=1.0 / pt.sqrt(omega_team_init * phi),
                shape=self.num_teams,
            )

            # Team strength random innovations
            team_innovation = pm.Normal(
                "team_innovation",
                mu=0.0,
                sigma=1.0,
                shape=(num_steps - 1, self.num_teams),
            )

            def evolve_team_strength(
                is_new_season_t,
                innovation_t,
                strength_prev,
                beta_season,
                beta_week,
                omega_season,
                omega_week,
                precision,
            ):
                """Evolve team strength for one time step."""
                # Select precision and regression based on season boundary
                omega_t = (
                    is_new_season_t * omega_season + (1 - is_new_season_t) * omega_week
                )
                beta_t = (
                    is_new_season_t * beta_season + (1 - is_new_season_t) * beta_week
                )

                # Mean-center previous strengths
                strength_centered = strength_prev - pt.mean(strength_prev)

                # Apply AR(1) evolution with innovation
                strength_new = beta_t * strength_centered + innovation_t / pt.sqrt(
                    omega_t * precision
                )
                return strength_new

            team_strength_sequence, _ = scan(
                fn=evolve_team_strength,
                sequences=[is_new_season_pt[1:], team_innovation],
                outputs_info=[team_strength_init],
                non_sequences=[
                    beta_team_season,
                    beta_team_week,
                    omega_team_season,
                    omega_team_week,
                    phi,
                ],
            )

            team_strength_all = pt.concatenate(
                [team_strength_init[None, :], team_strength_sequence], axis=0
            )
            pm.Deterministic("team_strength", team_strength_all)

            # ---------------------------------------------------
            # QB Strength Dynamics
            # ---------------------------------------------------
            # Initial QB strength precision
            omega_qb_init = pm.Gamma("omega_qb_init", alpha=0.5, beta=0.5 / 16)

            # Initial QB strength parameters
            experience_factor = pt.minimum(1.0, qb_experience_pt / 48.0)
            mu_qb_rookie = pm.Normal("mu_qb_rookie", mu=-2, sigma=1)
            mu_qb_veteran = pm.Normal("mu_qb_veteran", mu=0, sigma=1)
            mu_qb_by_experience = mu_qb_rookie + experience_factor * (
                mu_qb_veteran - mu_qb_rookie
            )

            # QB strength
            qb_strength_raw = pm.Normal(
                "qb_strength_raw",
                mu=mu_qb_by_experience,
                sigma=1.0 / pt.sqrt(omega_qb_init * phi),
                shape=self.num_qbs,
            )
            qb_strength = pm.Deterministic(
                "qb_strength", qb_strength_raw - pt.mean(qb_strength_raw)
            )

            # ---------------------------------------------------
            # Likelihood
            # ---------------------------------------------------
            # Pad design matrices and observations to same length
            max_games = max(x.shape[0] for x in team_mats)

            team_design_padded = []
            qb_design_padded = []
            y_padded = []
            mask = []

            for team_mat, qb_mat, y in zip(team_mats, qb_mats, y_obs):
                n_games = team_mat.shape[0]
                pad_size = max_games - n_games

                if pad_size > 0:
                    team_mat_padded = np.vstack(
                        [
                            team_mat,
                            np.zeros((pad_size, self.num_teams + 1), dtype=np.float32),
                        ]
                    )
                    qb_mat_padded = np.vstack(
                        [qb_mat, np.zeros((pad_size, self.num_qbs), dtype=np.float32)]
                    )
                    y_padded_week = np.concatenate(
                        [y, np.zeros(pad_size, dtype=np.float32)]
                    )
                    mask_week = np.concatenate(
                        [
                            np.ones(n_games, dtype=np.float32),
                            np.zeros(pad_size, dtype=np.float32),
                        ]
                    )
                else:
                    team_mat_padded = team_mat
                    qb_mat_padded = qb_mat
                    y_padded_week = y
                    mask_week = np.ones(n_games, dtype=np.float32)

                team_design_padded.append(team_mat_padded)
                qb_design_padded.append(qb_mat_padded)
                y_padded.append(y_padded_week)
                mask.append(mask_week)

            # Stack into 3D arrays
            team_design_array = np.array(team_design_padded, dtype=np.float32)
            qb_design_array = np.array(qb_design_padded, dtype=np.float32)
            y_array = np.array(y_padded, dtype=np.float32)
            mask_array = np.array(mask, dtype=np.float32)

            # Compute team component: X_team @ [team_strength; alpha_hfa]
            alpha_hfa_expanded = pt.tile(alpha_hfa, (num_steps, 1))
            team_params = pt.concatenate(
                [team_strength_all, alpha_hfa_expanded], axis=1
            )
            mu_team = pt.batched_dot(team_design_array, team_params[:, :, None])[
                :, :, 0
            ]

            # Compute QB component: X_qb @ qb_strength
            qb_params = pt.tile(qb_strength[None, :], (num_steps, 1))
            mu_qb = pt.batched_dot(qb_design_array, qb_params[:, :, None])[:, :, 0]

            # Total expected score differential
            mu_total = mu_team + mu_qb
            sigma_obs = 1.0 / pt.sqrt(phi)

            # Likelihood (only for non-padded observations)
            pm.Normal(
                "y_obs",
                mu=mu_total[mask_array == 1],
                sigma=sigma_obs,
                observed=y_array[mask_array == 1],
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
        alpha_hfa = self._get_posterior("alpha_hfa")
        beta_team_season = self._get_posterior("beta_team_season")
        beta_team_week = self._get_posterior("beta_team_week")
        team_strength = self._get_posterior("team_strength")
        team_strength_mean = np.mean(team_strength[:, :, -1, :], axis=1, keepdims=True)
        qb_strength = self._get_posterior("qb_strength")
        omega_qb = self._get_posterior("omega_qb_init")
        mu_qb_rookie = self._get_posterior("mu_qb_rookie")

        # Iterate through test data
        results = []
        for row in data.itertuples():

            # Get team indices
            home_team_idx = self.team_to_idx[row.home_team]
            away_team_idx = self.team_to_idx[row.away_team]

            # Compute evolved team strengths
            beta_team = beta_team_season if row.week == 1 else beta_team_week
            beta_team = beta_team[:, :, None]

            home_team_strength = beta_team * (
                team_strength[:, :, -1, home_team_idx : home_team_idx + 1]
                - team_strength_mean
            )
            away_team_strength = beta_team * (
                team_strength[:, :, -1, away_team_idx : away_team_idx + 1]
                - team_strength_mean
            )

            # Compute QB strengths
            if row.home_qb_id in self.qb_to_idx:
                home_qb_idx = self.qb_to_idx[row.home_qb_id]
                home_qb_strength = qb_strength[:, :, home_qb_idx : home_qb_idx + 1]
            else:
                prior_std = np.mean(1.0 / np.sqrt(omega_qb * phi))
                home_qb_strength = np.random.normal(
                    loc=mu_qb_rookie[:, :, None],
                    scale=prior_std,
                    size=(mu_qb_rookie.shape[0], mu_qb_rookie.shape[1], 1),
                )

            if row.away_qb_id in self.qb_to_idx:
                away_qb_idx = self.qb_to_idx[row.away_qb_id]
                away_qb_strength = qb_strength[:, :, away_qb_idx : away_qb_idx + 1]
            else:
                prior_std = np.mean(1.0 / np.sqrt(omega_qb * phi))
                away_qb_strength = np.random.normal(
                    loc=mu_qb_rookie[:, :, None],
                    scale=prior_std,
                    size=(mu_qb_rookie.shape[0], mu_qb_rookie.shape[1], 1),
                )

            # Compute home field advantage
            hfa = (1 - row.is_neutral) * alpha_hfa[:, :, None]

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

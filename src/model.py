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


class StateSpaceModel:
    """A state-space model for NFL team strength evolution."""

    def __init__(self):
        """Initialize model."""
        self.num_teams = 32
        self.team_to_idx = None
        self.trace = None
        self.model = None

    def prepare_data(self, data):
        """Prepare training data for model fitting.

        Args:
            data: NFL schedule data containing season, week, team names,
            location flags, and results.
        Returns:
            A tuple containing the following:
            - Weekly feature data.
            - Weekly observed point differentials.
            - Binary flags for season transitions.
            - Number of model steps (weeks).
        """
        train_data = data.copy()

        # Team indices
        teams = sorted(train_data["home_team"].unique())
        self.team_to_idx = {team: i for i, team in enumerate(teams)}
        train_data["home_idx"] = train_data["home_team"].map(self.team_to_idx)
        train_data["away_idx"] = train_data["away_team"].map(self.team_to_idx)
        # Season indices
        seasons = sorted(train_data["season"].unique())
        season_to_idx = {season: i for i, season in enumerate(seasons)}
        train_data["season_idx"] = train_data["season"].map(season_to_idx)
        # Week indices
        weeks = sorted(train_data["week"].unique())
        week_to_idx = {week: i for i, week in enumerate(weeks)}
        train_data["week_idx"] = train_data["week"].map(week_to_idx)
        # Step indices
        all_steps = train_data[["season_idx", "week_idx"]].drop_duplicates()
        all_steps["step_idx"] = np.arange(len(all_steps))
        train_data = pd.merge(train_data, all_steps, on=["season_idx", "week_idx"])
        # Build per-week design matrices and observed results
        num_steps = train_data["step_idx"].max()
        week_data = []
        for step in range(num_steps + 1):
            week_data.append(train_data[train_data["step_idx"] == step])
        is_new_season = (
            np.array([wd["week_idx"].iloc[0] for wd in week_data]) == 0
        ).astype("int32")
        x_mats = [self.build_design_matrix(week_data[i]) for i in range(num_steps + 1)]
        y_obs = [week_data[i]["result"].values for i in range(num_steps + 1)]

        return x_mats, y_obs, is_new_season, num_steps

    def build_design_matrix(self, week_data):
        """Build design matrix for a single week of games.

        Args:
            week_data: Single week feature data.
        Returns:
            A matrix encoding the week's features.
        """
        design_matrix = np.zeros((len(week_data), self.num_teams + 1))
        for i, row in enumerate(week_data.itertuples()):
            home = int(row.home_idx)
            away = int(row.away_idx)
            is_home = int(1 - row.is_neutral)
            design_matrix[i, home] = 1
            design_matrix[i, away] = -1
            design_matrix[i, self.num_teams] = is_home
        return design_matrix

    def build_model(self, x_mats, y_obs, is_new_season, num_steps):
        """Build the full PyMC state-space model."""
        with pm.Model() as model:
            # ---------------------------------------------------
            # Priors
            # ---------------------------------------------------
            phi = pm.Gamma("phi", alpha=0.5, beta=0.5 * 100)
            omega_s = pm.Gamma("omega_s", alpha=0.5, beta=0.5 / 16)
            omega_w = pm.Gamma("omega_w", alpha=0.5, beta=0.5 / 60)
            beta_s = pm.Normal("beta_s", mu=0.98, sigma=1)
            beta_w = pm.Normal("beta_w", mu=0.995, sigma=1)
            omega_zero = pm.Gamma("omega_0", alpha=0.5, beta=0.5 / 6)
            alpha = pm.Normal("alpha", mu=2, sigma=1)

            # ---------------------------------------------------
            # State Evolution
            # ---------------------------------------------------
            # Initial team abilities
            theta_init = pm.Normal(
                "theta_0",
                mu=0,
                sigma=1 / pt.sqrt(omega_zero * phi),
                shape=self.num_teams,
            )

            # Innovation noise for all time steps
            innovation_noise = pm.Normal(
                "innovation_noise", mu=0, sigma=1, shape=(num_steps, self.num_teams)
            )

            # State evolution using scan
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

            # Concatenate initial theta with evolved sequence
            all_theta = pt.concatenate([theta_init[None, :], theta_sequence], axis=0)
            pm.Deterministic("theta", all_theta)

            # ---------------------------------------------------
            # Likelihood
            # ---------------------------------------------------
            # Stack all design matrices and observations
            max_games = max(x.shape[0] for x in x_mats)

            # Pad design matrices and observations to same length
            x_padded = []
            y_padded = []
            mask = []

            for x, y in zip(x_mats, y_obs):
                n_games = x.shape[0]
                if n_games < max_games:
                    x_pad = np.vstack(
                        [x, np.zeros((max_games - n_games, self.num_teams + 1))]
                    )
                    y_pad = np.concatenate([y, np.zeros(max_games - n_games)])
                    m = np.concatenate(
                        [np.ones(n_games), np.zeros(max_games - n_games)]
                    )
                else:
                    x_pad = x
                    y_pad = y
                    m = np.ones(n_games)
                x_padded.append(x_pad)
                y_padded.append(y_pad)
                mask.append(m)

            x_all = np.array(x_padded)
            y_all = np.array(y_padded)
            mask_all = np.array(mask)

            # Combine theta and alpha for prediction
            alpha_column = pt.tile(alpha, (num_steps + 1, 1))
            theta_expanded = pt.concatenate([all_theta, alpha_column], axis=1)

            # Compute predictions: x_all @ theta_expanded.T, then take diagonal
            mu_all = pt.batched_dot(x_all, theta_expanded[:, :, None])[:, :, 0]
            sigma_step = 1 / pm.math.sqrt(phi)

            # Likelihood with mask
            pm.Normal(
                "y_obs",
                mu=mu_all[mask_all == 1],
                sigma=sigma_step,
                observed=y_all[mask_all == 1],
            )

        return model

    def fit(self, data):
        """Fit the state-space model to historical game data."""
        # Prepare data
        x_mats, y_obs, is_new_season, num_steps = self.prepare_data(data)

        # Build model
        self.model = self.build_model(x_mats, y_obs, is_new_season, num_steps)

        # Sample
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

    def predict(self, data):
        """Generate predictions with full Bayesian uncertainty.

        Returns per-game win probabilities, expected score differential,
        and percentile-based confidence intervals for both teams.

        Args:
            data: Test data.
        Returns:
            A DataFrame with prediction results including columns for
            means, standard deviations, and percentiles (1–99).
        """
        if self.trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        theta = self.trace.posterior["theta"].values
        alpha = self.trace.posterior["alpha"].values
        beta_s = self.trace.posterior["beta_s"].values
        beta_w = self.trace.posterior["beta_w"].values

        mean_theta = np.mean(theta[:, :, -1, :], axis=-1, keepdims=True)

        percentiles = list(range(1, 100))

        results = []

        for row in data.itertuples():
            home_team_idx = self.team_to_idx[row.home_team]
            away_team_idx = self.team_to_idx[row.away_team]
            beta = beta_s if row.week == 1 else beta_w
            beta = beta[:, :, None]

            home_team_strength = beta * (
                theta[:, :, -1, home_team_idx : home_team_idx + 1] - mean_theta
            )
            away_team_strength = beta * (
                theta[:, :, -1, away_team_idx : away_team_idx + 1] - mean_theta
            )
            hfa = (
                np.zeros_like(alpha)[:, :, None]
                if row.is_neutral
                else alpha[:, :, None]
            )

            prediction = home_team_strength - away_team_strength + hfa

            pred_percentiles = np.percentile(prediction, percentiles)
            home_percentiles = np.percentile(home_team_strength, percentiles)
            away_percentiles = np.percentile(away_team_strength, percentiles)

            results.append(
                [
                    row.home_team,
                    row.away_team,
                    float((prediction > 0).mean()),
                    float(prediction.mean()),
                    float(prediction.std()),
                    *pred_percentiles,
                    float(home_team_strength.mean()),
                    float(home_team_strength.std()),
                    *home_percentiles,
                    float(away_team_strength.mean()),
                    float(away_team_strength.std()),
                    *away_percentiles,
                ]
            )

        columns = (
            [
                "home_team",
                "away_team",
                "home_win_prob",
                "prediction_mean",
                "prediction_std",
            ]
            + [f"prediction_ci_{p:02d}" for p in percentiles]
            + ["home_strength_mean", "home_strength_std"]
            + [f"home_strength_ci_{p:02d}" for p in percentiles]
            + ["away_strength_mean", "away_strength_std"]
            + [f"away_strength_ci_{p:02d}" for p in percentiles]
        )

        return pd.DataFrame(results, columns=columns)

"""State space model implementation."""

import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor import scan

warnings.filterwarnings(
    "ignore",
    message="batched_dot is deprecated",
    category=FutureWarning,
    module="pytensor.tensor.blas",
)


class StateSpaceModel:
    """Bayesian state-space model for NFL team strength evolution."""

    def __init__(self):
        """Initialize model."""
        self.num_teams = 32
        self.team_to_idx = None
        self.trace = None
        self.model = None

    def prepare_data(self, data):
        """Prepare training data for model fitting."""
        train_data = data.copy()

        # Team index
        teams = sorted(train_data["home_team"].unique())
        self.team_to_idx = {team: i for i, team in enumerate(teams)}
        train_data["home_idx"] = train_data["home_team"].map(self.team_to_idx)
        train_data["away_idx"] = train_data["away_team"].map(self.team_to_idx)
        # Season index
        seasons = sorted(train_data["season"].unique())
        season_to_idx = {season: i for i, season in enumerate(seasons)}
        train_data["season_idx"] = train_data["season"].map(season_to_idx)
        # Week index
        weeks = sorted(train_data["week"].unique())
        week_to_idx = {week: i for i, week in enumerate(weeks)}
        train_data["week_idx"] = train_data["week"].map(week_to_idx)
        # Step index
        all_steps = train_data[["season_idx", "week_idx"]].drop_duplicates()
        all_steps["step_idx"] = np.arange(len(all_steps))
        train_data = pd.merge(train_data, all_steps, on=["season_idx", "week_idx"])

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
        "Build design matrix for a single week of games."
        design_matrix = np.zeros((len(week_data), self.num_teams + 1))
        for i, row in enumerate(week_data.itertuples()):
            home = int(row.home_idx)
            away = int(row.away_idx)
            is_home = int(1 - row.is_neutral)
            design_matrix[i, home] = 1
            design_matrix[i, away] = -1
            design_matrix[i, self.num_teams] = is_home
        return design_matrix

    # pylint: disable=too-many-locals
    def build_model(self, x_mats, y_obs, is_new_season, num_steps):
        """Build the PyMC model."""
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
            alpha = pm.Normal("alpha", mu=3, sigma=1)

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
            # pylint: disable=too-many-arguments, too-many-positional-arguments
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
            # pylint: disable=unsubscriptable-object
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
        """Fit model on training data."""
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

    def get_parameters(self):
        """Get most recent team strengths, home field advantage, and regression
        parameters from the fit model."""
        if self.trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        theta = self.trace.posterior["theta"].mean(dim=["chain", "draw"]).values[-1, :]
        alpha = self.trace.posterior["alpha"].mean().item()
        beta_s = self.trace.posterior["beta_s"].mean().item()
        beta_w = self.trace.posterior["beta_w"].mean().item()

        return theta, alpha, beta_s, beta_w

    def predict(self, data):
        """Predict games in testing data."""
        theta, alpha, beta_s, beta_w = self.get_parameters()

        results = []

        for row in data.itertuples():
            home_team_idx = self.team_to_idx[row.home_team]
            away_team_idx = self.team_to_idx[row.away_team]

            if row.week == 1:
                beta = beta_s
            else:
                beta = beta_w
            home_team_strength = beta * (theta[home_team_idx] - np.mean(theta))
            away_team_strength = beta * (theta[away_team_idx] - np.mean(theta))

            if row.is_neutral:
                alpha_game = 0
            else:
                alpha_game = alpha
            prediction = home_team_strength - away_team_strength + alpha_game

            results.append(
                [
                    row.home_team,
                    row.away_team,
                    home_team_strength,
                    away_team_strength,
                    prediction,
                ]
            )

        return pd.DataFrame(
            results,
            columns=[
                "home_team",
                "away_team",
                "home_strength",
                "away_strength",
                "prediction",
            ],
        )

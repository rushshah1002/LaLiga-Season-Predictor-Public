from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

# log(lambda_home) = mu + home_adv + attack[home] + defense[away]
# log(lambda_away) = mu + attack[away] + defense[home]
# attack/defense centered to sum to zero for identifiability

@dataclass
class PoissonStrengthModel:
    reg: float = 1.0
    home_adv: float = 0.10
    fitted_: bool = False

    teams_: List[str] = None
    params_: np.ndarray = None  # [mu, home_adv, attacks..., defenses...]

    def _pack(self, mu, home_adv, attack, defense):
        return np.concatenate([[mu, home_adv], attack, defense])

    def _unpack(self, x, n_teams):
        mu = x[0]
        ha = x[1]
        attack = x[2:2+n_teams]
        defense = x[2+n_teams:2+2*n_teams]
        attack = attack - attack.mean()
        defense = defense - defense.mean()
        return mu, ha, attack, defense

    def fit(self, df: pd.DataFrame) -> "PoissonStrengthModel":
        df = df.copy().sort_values("date")
        teams = sorted(set(df["home_team"]).union(set(df["away_team"])))
        team_to_idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        home_idx = df["home_team"].map(team_to_idx).to_numpy()
        away_idx = df["away_team"].map(team_to_idx).to_numpy()
        y_home = df["home_goals"].to_numpy().astype(int)
        y_away = df["away_goals"].to_numpy().astype(int)

        mu0 = np.log(np.maximum(df[["home_goals", "away_goals"]].to_numpy().mean(), 1e-6))
        x0 = self._pack(mu0, self.home_adv, np.zeros(n), np.zeros(n))

        def nll(x):
            mu, ha, a, d = self._unpack(x, n)
            lam_home = np.exp(mu + ha + a[home_idx] + d[away_idx])
            lam_away = np.exp(mu + a[away_idx] + d[home_idx])

            ll = (
                -lam_home + y_home * np.log(lam_home) - gammaln(y_home + 1) +
                -lam_away + y_away * np.log(lam_away) - gammaln(y_away + 1)
            ).sum()
            reg = self.reg * (np.sum(a * a) + np.sum(d * d))
            return -ll + reg

        res = minimize(nll, x0, method="L-BFGS-B")
        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")

        self.teams_ = teams
        self.params_ = res.x
        self.fitted_ = True
        return self

    def predict_lambdas(self, home_team: str, away_team: str) -> Tuple[float, float]:
        if not self.fitted_:
            raise ValueError("Model not fitted")
        teams = self.teams_
        n = len(teams)
        idx = {t: i for i, t in enumerate(teams)}
        mu, ha, a, d = self._unpack(self.params_, n)
        hi = idx[home_team]
        ai = idx[away_team]
        lam_home = float(np.exp(mu + ha + a[hi] + d[ai]))
        lam_away = float(np.exp(mu + a[ai] + d[hi]))
        return lam_home, lam_away

def poisson_match_probs(lam_home: float, lam_away: float, max_goals: int = 10) -> np.ndarray:
    from scipy.stats import poisson
    ph = pd_ = pa = 0.0
    for i in range(max_goals + 1):
        pi = poisson.pmf(i, lam_home)
        for j in range(max_goals + 1):
            p = pi * poisson.pmf(j, lam_away)
            if i > j:
                ph += p
            elif i == j:
                pd_ += p
            else:
                pa += p
    s = ph + pd_ + pa
    return np.array([ph / s, pd_ / s, pa / s], dtype=float)

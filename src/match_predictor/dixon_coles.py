import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson as poisson_dist


class DixonColesModel:
    """
    Time-weighted Dixon-Coles Poisson model for scoreline prediction.

    Fits per-team attack/defense strengths plus a home advantage multiplier
    and a low-score correction (rho) via maximum likelihood. Recent matches
    are up-weighted by exp(-xi * days_ago) so current form matters more.
    """

    def __init__(self, xi=0.0065):
        self.xi = xi
        self.attack = {}
        self.defense = {}
        self.home_adv = None
        self.rho = None
        self.teams = None

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------

    def fit(self, df):
        df = df.dropna(subset=["home_goals", "away_goals"]).copy()
        df["home_goals"] = df["home_goals"].astype(int)
        df["away_goals"] = df["away_goals"].astype(int)
        df = df.sort_values("date").reset_index(drop=True)

        self.teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        n = len(self.teams)
        team_idx = {t: i for i, t in enumerate(self.teams)}

        ref_date = df["date"].max()
        days_ago = (ref_date - df["date"]).dt.days.values
        weights = np.exp(-self.xi * days_ago)

        hi = df["home_team"].map(team_idx).values
        ai = df["away_team"].map(team_idx).values
        hg = df["home_goals"].values
        ag = df["away_goals"].values

        def neg_ll(params):
            # log-parameterize attack/defense/home_adv for positivity
            log_att = np.zeros(n)
            log_att[1:] = params[: n - 1]
            log_def = params[n - 1 : 2 * n - 1]
            log_ha = params[2 * n - 1]
            rho = params[2 * n]

            att = np.exp(log_att)
            defe = np.exp(log_def)
            ha = np.exp(log_ha)

            lam_h = att[hi] * defe[ai] * ha
            lam_a = att[ai] * defe[hi]

            log_p = poisson_dist.logpmf(hg, lam_h) + poisson_dist.logpmf(ag, lam_a)

            # Dixon-Coles correction for 0-0, 1-0, 0-1, 1-1
            tau = np.ones(len(df))
            tau[(hg == 0) & (ag == 0)] = np.clip(
                1 - lam_h[(hg == 0) & (ag == 0)] * lam_a[(hg == 0) & (ag == 0)] * rho, 1e-10, None
            )
            tau[(hg == 1) & (ag == 0)] = np.clip(1 + lam_a[(hg == 1) & (ag == 0)] * rho, 1e-10, None)
            tau[(hg == 0) & (ag == 1)] = np.clip(1 + lam_h[(hg == 0) & (ag == 1)] * rho, 1e-10, None)
            tau[(hg == 1) & (ag == 1)] = np.clip(1 - rho, 1e-10, None)

            return -(weights * (log_p + np.log(tau))).sum()

        x0 = np.zeros(2 * n + 1)
        x0[2 * n - 1] = np.log(1.3)  # start: home adv ~1.3x

        bounds = (
            [(-3.0, 3.0)] * (n - 1)       # log attack (team 0 fixed at 0)
            + [(-3.0, 3.0)] * n            # log defense
            + [(np.log(0.5), np.log(3.0))] # log home advantage
            + [(-1.0, 1.0)]                # rho
        )

        result = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 1000, "ftol": 1e-10})

        params = result.x
        log_att = np.zeros(n)
        log_att[1:] = params[: n - 1]
        att = np.exp(log_att)
        defe = np.exp(params[n - 1 : 2 * n - 1])

        # normalize so mean attack = 1; multiply defense by att_mean to preserve lambda
        att_mean = att.mean()
        att /= att_mean
        defe *= att_mean

        self.attack = dict(zip(self.teams, att))
        self.defense = dict(zip(self.teams, defe))
        self.home_adv = float(np.exp(params[2 * n - 1]))
        self.rho = float(params[2 * n])
        return self

    # ------------------------------------------------------------------
    # prediction
    # ------------------------------------------------------------------

    def _lambdas(self, home_team, away_team):
        lam_h = self.attack[home_team] * self.defense[away_team] * self.home_adv
        lam_a = self.attack[away_team] * self.defense[home_team]
        return lam_h, lam_a

    def predict_scoreline(self, home_team, away_team, max_goals=8):
        """NxN probability matrix where matrix[h, a] = P(home scores h, away scores a)."""
        lam_h, lam_a = self._lambdas(home_team, away_team)

        h_probs = poisson_dist.pmf(np.arange(max_goals + 1), lam_h)
        a_probs = poisson_dist.pmf(np.arange(max_goals + 1), lam_a)
        matrix = np.outer(h_probs, a_probs)

        # apply DC correction
        rho = self.rho
        matrix[0, 0] = max(matrix[0, 0] * (1 - lam_h * lam_a * rho), 1e-10)
        matrix[1, 0] = max(matrix[1, 0] * (1 + lam_a * rho), 1e-10)
        matrix[0, 1] = max(matrix[0, 1] * (1 + lam_h * rho), 1e-10)
        matrix[1, 1] = max(matrix[1, 1] * (1 - rho), 1e-10)

        return matrix / matrix.sum()

    def predict_result(self, home_team, away_team, max_goals=8):
        """Returns {'HOME_TEAM': p, 'DRAW': p, 'AWAY_TEAM': p}."""
        m = self.predict_scoreline(home_team, away_team, max_goals)
        return {
            "HOME_TEAM": float(np.tril(m, -1).sum()),
            "DRAW":      float(np.trace(m)),
            "AWAY_TEAM": float(np.triu(m, 1).sum()),
        }

    def top_scorelines(self, home_team, away_team, n=5, max_goals=8):
        """Returns list of (home_goals, away_goals, probability) sorted by probability desc."""
        m = self.predict_scoreline(home_team, away_team, max_goals)
        flat_idx = np.argsort(m, axis=None)[::-1][:n]
        rows, cols = np.unravel_index(flat_idx, m.shape)
        return [(int(h), int(a), float(m[h, a])) for h, a in zip(rows, cols)]

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------

    def team_strengths(self):
        """DataFrame of attack / defense ratings, sorted by attack strength."""
        return pd.DataFrame(
            {"attack": self.attack, "defense": self.defense}
        ).sort_values("attack", ascending=False).round(4)

    def evaluate(self, df, max_goals=8):
        """
        Runs the fitted model on df and reports:
          - accuracy of most-likely result vs actual
          - ranked probability score (RPS) — lower is better
        """
        order = ["HOME_TEAM", "DRAW", "AWAY_TEAM"]
        rows = []
        for _, r in df.iterrows():
            if r["home_team"] not in self.attack or r["away_team"] not in self.attack:
                continue
            res = self.predict_result(r["home_team"], r["away_team"], max_goals)
            probs = [res["HOME_TEAM"], res["DRAW"], res["AWAY_TEAM"]]
            pred = order[int(np.argmax(probs))]
            rows.append({
                "actual": r["result"],
                "predicted": pred,
                "p_home": probs[0],
                "p_draw": probs[1],
                "p_away": probs[2],
            })

        results = pd.DataFrame(rows)
        acc = (results["actual"] == results["predicted"]).mean()

        # ranked probability score
        actual_enc = pd.get_dummies(results["actual"]).reindex(columns=order, fill_value=0)
        pred_probs = results[["p_home", "p_draw", "p_away"]].values
        rps = _rps(actual_enc.values, pred_probs)

        return acc, rps, results


def _rps(y_onehot, probs):
    """Mean ranked probability score (multiclass), lower = better."""
    cum_prob = np.cumsum(probs, axis=1)
    cum_actual = np.cumsum(y_onehot, axis=1)
    return float(np.mean(np.sum((cum_prob - cum_actual) ** 2, axis=1) / (probs.shape[1] - 1)))

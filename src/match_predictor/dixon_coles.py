import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson as poisson_dist

from match_predictor.evaluation import RESULT_ORDER, rps


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

        is_00 = (hg == 0) & (ag == 0)
        is_10 = (hg == 1) & (ag == 0)
        is_01 = (hg == 0) & (ag == 1)
        is_11 = (hg == 1) & (ag == 1)
        # log(hg!) + log(ag!) is constant in the parameters
        log_fact = gammaln(hg + 1) + gammaln(ag + 1)

        def neg_ll_and_grad(params):
            # log-parameterize attack/defense/home_adv for positivity
            log_att = np.zeros(n)
            log_att[1:] = params[: n - 1]
            log_def = params[n - 1 : 2 * n - 1]
            log_ha = params[2 * n - 1]
            rho = params[2 * n]

            lam_h = np.exp(log_att[hi] + log_def[ai] + log_ha)
            lam_a = np.exp(log_att[ai] + log_def[hi])

            log_p = hg * np.log(lam_h) - lam_h + ag * np.log(lam_a) - lam_a - log_fact

            # Dixon-Coles correction for 0-0, 1-0, 0-1, 1-1, with its partial
            # derivatives w.r.t. log(lam_h), log(lam_a) and rho
            tau_raw = np.ones(len(hg))
            tau_raw[is_00] = 1 - lam_h[is_00] * lam_a[is_00] * rho
            tau_raw[is_10] = 1 + lam_a[is_10] * rho
            tau_raw[is_01] = 1 + lam_h[is_01] * rho
            tau_raw[is_11] = 1 - rho
            tau = np.clip(tau_raw, 1e-10, None)

            dtau_h = np.zeros(len(hg))
            dtau_a = np.zeros(len(hg))
            dtau_rho = np.zeros(len(hg))
            dtau_h[is_00] = dtau_a[is_00] = -lam_h[is_00] * lam_a[is_00] * rho
            dtau_rho[is_00] = -lam_h[is_00] * lam_a[is_00]
            dtau_a[is_10] = lam_a[is_10] * rho
            dtau_rho[is_10] = lam_a[is_10]
            dtau_h[is_01] = lam_h[is_01] * rho
            dtau_rho[is_01] = lam_h[is_01]
            dtau_rho[is_11] = -1.0
            # clipped rows are flat
            live = (tau_raw > 1e-10) / tau
            dtau_h *= live
            dtau_a *= live
            dtau_rho *= live

            g_h = weights * (hg - lam_h + dtau_h)   # d loglik / d log(lam_h)
            g_a = weights * (ag - lam_a + dtau_a)   # d loglik / d log(lam_a)

            grad_att = np.bincount(hi, g_h, n) + np.bincount(ai, g_a, n)
            grad_def = np.bincount(ai, g_h, n) + np.bincount(hi, g_a, n)
            grad = np.concatenate([
                grad_att[1:], grad_def, [g_h.sum()], [(weights * dtau_rho).sum()],
            ])

            nll = -(weights * (log_p + np.log(tau))).sum()
            return nll, -grad

        x0 = np.zeros(2 * n + 1)
        x0[2 * n - 1] = np.log(1.3)  # start: home adv ~1.3x

        bounds = (
            [(-3.0, 3.0)] * (n - 1)       # log attack (team 0 fixed at 0)
            + [(-3.0, 3.0)] * n            # log defense
            + [(np.log(0.5), np.log(3.0))] # log home advantage
            + [(-1.0, 1.0)]                # rho
        )

        result = minimize(neg_ll_and_grad, x0, jac=True, method="L-BFGS-B", bounds=bounds,
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
        order = RESULT_ORDER
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
        score = rps(actual_enc.values, pred_probs)

        return acc, score, results

    def set_rating(self, team, attack, defense):
        self.attack[team] = attack
        self.defense[team] = defense


def backtest(df, start_date, xi=0.0065, window_days=1095, refit_days=7,
             min_matches=5, max_goals=8):
    """
    Out-of-sample walk-forward evaluation. Every `refit_days` the model is
    refitted on matches strictly before that date (within `window_days`) and
    used to predict the next block. Teams with fewer than `min_matches` in the
    window (i.e. newly promoted sides) get a promoted-team rating: the average
    of the three weakest sides in the fitted model.

    Returns a DataFrame (same index as df, rows from start_date on) with
    p_home, p_draw, p_away and the actual result.
    """
    df = df.dropna(subset=["home_goals", "away_goals"]).sort_values("date")
    start_date = pd.Timestamp(start_date)
    end_date = df["date"].max()

    frames = []
    block_start = start_date
    while block_start <= end_date:
        block_end = block_start + pd.Timedelta(days=refit_days)
        block = df[(df["date"] >= block_start) & (df["date"] < block_end)]
        block_start, window_start = block_end, block_start - pd.Timedelta(days=window_days)
        if block.empty:
            continue

        train = df[(df["date"] < block["date"].min()) & (df["date"] >= window_start)]
        model = DixonColesModel(xi=xi).fit(train)
        _fill_sparse_teams(model, train, block, min_matches)

        probs = [model.predict_result(h, a, max_goals)
                 for h, a in zip(block["home_team"], block["away_team"])]
        frames.append(pd.DataFrame({
            "date":      block["date"],
            "home_team": block["home_team"],
            "away_team": block["away_team"],
            "result":    block["result"],
            "p_home":    [p["HOME_TEAM"] for p in probs],
            "p_draw":    [p["DRAW"] for p in probs],
            "p_away":    [p["AWAY_TEAM"] for p in probs],
        }, index=block.index))

    return pd.concat(frames)


def _fill_sparse_teams(model, train, upcoming, min_matches):
    counts = pd.concat([train["home_team"], train["away_team"]]).value_counts()
    established = [t for t in model.teams if counts.get(t, 0) >= min_matches]

    # weakest = lowest attack / defense ratio (defense is goals-conceded multiplier)
    strength = sorted(established, key=lambda t: model.attack[t] / model.defense[t])
    weakest = strength[:3]
    prom_att = float(np.mean([model.attack[t] for t in weakest]))
    prom_def = float(np.mean([model.defense[t] for t in weakest]))

    for team in set(upcoming["home_team"]) | set(upcoming["away_team"]):
        if counts.get(team, 0) < min_matches:
            model.set_rating(team, prom_att, prom_def)

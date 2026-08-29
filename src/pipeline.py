from __future__ import annotations
import numpy as np
import pandas as pd

from .data_loader import load_la_liga_matches, LoadSpec
from .odds import add_market_probs
from .model import PoissonStrengthModel, poisson_match_probs
from .evaluation import add_true_outcome, summarize_probs, best_ev_bet

# train: 1920-2223, val: 2324, test: 2425, future: 2526 (if present)

def split_data(df: pd.DataFrame) -> dict:
    train = df[df["season"].isin(["1920", "2021", "2122", "2223"])].copy()
    val = df[df["season"] == "2324"].copy()
    test = df[df["season"] == "2425"].copy()
    future = df[df["season"] == "2526"].copy() if "2526" in df["season"].values else pd.DataFrame()
    return {"train": train, "val": val, "test": test, "future": future}

def predict_matches(model: PoissonStrengthModel, matches_df: pd.DataFrame) -> pd.DataFrame:
    probs = []
    for r in matches_df.itertuples(index=False):
        try:
            lam_h, lam_a = model.predict_lambdas(r.home_team, r.away_team)
            probs.append(poisson_match_probs(lam_h, lam_a, max_goals=10))
        except KeyError:
            probs.append([np.nan, np.nan, np.nan])

    probs = np.vstack(probs)
    result_df = matches_df.copy()
    result_df["p_home_model"] = probs[:, 0]
    result_df["p_draw_model"] = probs[:, 1]
    result_df["p_away_model"] = probs[:, 2]

    if "odds_home" in result_df.columns:
        evs = []
        for i, r in enumerate(result_df.itertuples(index=False)):
            if pd.notna(r.odds_home):
                odds_triplet = np.array([r.odds_home, r.odds_draw, r.odds_away])
                _, ev = best_ev_bet(probs[i], odds_triplet)
                evs.append(ev)
            else:
                evs.append(np.nan)
        result_df["best_ev"] = evs

    return result_df

def run() -> dict:
    df = load_la_liga_matches(LoadSpec(league="la_liga"))
    df = add_market_probs(df)
    df = add_true_outcome(df)
    print(f"{len(df)} matches, seasons {sorted(df['season'].unique())}")

    splits = split_data(df)
    train_df, val_df, test_df, future_df = splits["train"], splits["val"], splits["test"], splits["future"]
    print(f"train {len(train_df)} / val {len(val_df)} / test {len(test_df)}", end="")
    if len(future_df) > 0:
        print(f" / future {len(future_df)}")
    else:
        print()

    train_combined = pd.concat([train_df, val_df], ignore_index=True)
    model = PoissonStrengthModel(reg=1.0, home_adv=0.10).fit(train_combined)

    test_df = predict_matches(model, test_df)
    test_df_complete = test_df.dropna(subset=["p_home_model"])

    if len(test_df_complete) > 0:
        model_probs = test_df_complete[["p_home_model", "p_draw_model", "p_away_model"]].values
        market_probs = test_df_complete[["market_p_home", "market_p_draw", "market_p_away"]].values
        y_true = test_df_complete["y_true"].values
        model_metrics = summarize_probs(model_probs, y_true)
        market_metrics = summarize_probs(market_probs, y_true)
    else:
        model_metrics = market_metrics = None

    future_results = None
    if len(future_df) > 0:
        future_results = predict_matches(model, future_df)

    return {
        "model": model,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "test_metrics": {"model": model_metrics, "market": market_metrics},
        "future_df": future_results,
    }

if __name__ == "__main__":
    results = run()
    m = results["test_metrics"]["model"]
    mk = results["test_metrics"]["market"]

    if m:
        print(f"\nmodel   log_loss {m['log_loss']:.4f}  brier {m['brier']:.4f}")
        print(f"market  log_loss {mk['log_loss']:.4f}  brier {mk['brier']:.4f}")
        print(f"avg ev  {results['test_df']['best_ev'].mean():.4f}")

    if results["future_df"] is not None and len(results["future_df"]) > 0:
        sample = results["future_df"][["date", "home_team", "away_team",
                                        "p_home_model", "p_draw_model", "p_away_model"]].head(5)
        print(f"\n{len(results['future_df'])} future matches")
        print(sample.to_string(index=False))

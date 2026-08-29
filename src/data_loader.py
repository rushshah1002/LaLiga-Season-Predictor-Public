from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

# Source: football-data.co.uk historical CSVs

FOOTBALL_DATA_BASE = "https://www.football-data.co.uk/mmz4281"
LEAGUE_CODES = {"la_liga": "SP1"}
DEFAULT_SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425"]

@dataclass(frozen=True)
class LoadSpec:
    league: str = "la_liga"
    seasons: List[str] = None

    def __post_init__(self):
        if self.seasons is None:
            object.__setattr__(self, "seasons", DEFAULT_SEASONS)

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
        "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
    }
    df = df.rename(columns=mapping)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    return df

def _compute_avg_1x2_odds(df: pd.DataFrame) -> pd.DataFrame:
    # prefer AvgH/AvgD/AvgA, else average across known bookmaker columns
    if {"AvgH", "AvgD", "AvgA"}.issubset(df.columns):
        df["odds_home"] = pd.to_numeric(df["AvgH"], errors="coerce")
        df["odds_draw"] = pd.to_numeric(df["AvgD"], errors="coerce")
        df["odds_away"] = pd.to_numeric(df["AvgA"], errors="coerce")
        return df

    home_cols = [c for c in df.columns if c.endswith("H") and len(c) <= 6]
    draw_cols = [c for c in df.columns if c.endswith("D") and len(c) <= 6]
    away_cols = [c for c in df.columns if c.endswith("A") and len(c) <= 6]

    def _is_odds_col(c: str) -> bool:
        return any(c.startswith(p) for p in ["B365", "BW", "IW", "PS", "WH", "VC", "Max", "Avg", "SB", "SJ", "SY", "GB", "LB", "BB", "BM"])

    home_cols = [c for c in home_cols if _is_odds_col(c)]
    draw_cols = [c for c in draw_cols if _is_odds_col(c)]
    away_cols = [c for c in away_cols if _is_odds_col(c)]

    if not (home_cols and draw_cols and away_cols):
        raise ValueError("Could not find 1X2 odds columns. Consider using seasons with AvgH/AvgD/AvgA columns.")

    df["odds_home"] = pd.to_numeric(df[home_cols].mean(axis=1), errors="coerce")
    df["odds_draw"] = pd.to_numeric(df[draw_cols].mean(axis=1), errors="coerce")
    df["odds_away"] = pd.to_numeric(df[away_cols].mean(axis=1), errors="coerce")
    return df

def load_la_liga_matches(spec: Optional[LoadSpec] = None) -> pd.DataFrame:
    if spec is None:
        spec = LoadSpec()
    code = LEAGUE_CODES.get(spec.league)
    if code is None:
        raise ValueError(f"Unknown league '{spec.league}'. Known: {list(LEAGUE_CODES.keys())}")

    frames = []
    for season in spec.seasons:
        url = f"{FOOTBALL_DATA_BASE}/{season}/{code}.csv"
        df = pd.read_csv(url)
        df = _standardize_columns(df)
        df = _compute_avg_1x2_odds(df)
        df["season"] = season
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals", "odds_home", "odds_draw", "odds_away"])
    out = out.sort_values("date").reset_index(drop=True)

    out["home_goals"] = pd.to_numeric(out["home_goals"], errors="coerce")
    out["away_goals"] = pd.to_numeric(out["away_goals"], errors="coerce")
    out = out.dropna(subset=["home_goals", "away_goals"]).reset_index(drop=True)

    return out

"""
Project: Does heavier recent workload hurt NBA player performance?

Each row = one NBA player in one game.
  Explanatory (x): minutes played in the player's PREVIOUS 5 games (same season)
  Response    (y): Game Score (GmSc) in the current game

Model:    GmSc = beta_0 + beta_1 * MP_prev5 + eps
H0:       beta_1 = 0          (no relationship)
HA:       beta_1 < 0          (heavier recent workload -> worse performance)
Test:     one-sided t-test on the OLS slope.

Data: scraped from basketball-reference.com player game logs for a curated set
of high-minute, durable players across the past few seasons. Results cached to
local CSVs so re-runs are instant.
"""
import io
import sys
import time
from pathlib import Path

# Player names contain non-ASCII chars (Jokić, Dončić, ...); Windows console
# defaults to cp1252 and crashes on print(). Force UTF-8 + replace fallback.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from scipy import stats
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
CACHE = ROOT / "bbref_cache"
CACHE.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (educational research project)"}
SEASONS = [2022, 2023, 2024]      # season-END years => 2021-22, 2022-23, 2023-24
WINDOW = 5                          # rolling minutes window (previous N games)
MIN_MPG = 20.0                      # qualifying player must average >= MIN_MPG
MIN_GAMES_SEASON = 60               # ...and play in >= MIN_GAMES_SEASON games
SLEEP_SEC = 4.0                     # be polite to basketball-reference


def discover_qualifying_players(season: int) -> list[dict]:
    """
    Return [{'player_id', 'name', 'G', 'MPG'}, ...] for players who averaged
    >= MIN_MPG and played in >= MIN_GAMES_SEASON games during this season.
    Result is cached per season.
    """
    cache_file = CACHE / f"_qualifiers_{season}.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file).to_dict("records")

    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_per_game.html"
    r = requests.get(url, headers=HEADERS, timeout=30)
    time.sleep(SLEEP_SEC)
    soup = BeautifulSoup(r.text, "lxml")
    table = soup.find("table", id="per_game_stats")
    if table is None:
        raise RuntimeError(f"per_game_stats table not found for {season}")

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        if "thead" in (tr.get("class") or []):
            continue
        link = tr.find("a", href=lambda h: h and h.startswith("/players/"))
        if link is None:
            continue
        pid = link["href"].split("/")[-1].replace(".html", "")
        name = link.get_text(strip=True)
        cells = {c.get("data-stat"): c for c in tr.find_all(["th", "td"])}
        try:
            g = int(cells["games"].get_text(strip=True))
            mpg = float(cells["mp_per_g"].get_text(strip=True))
        except (KeyError, ValueError):
            continue
        rows.append({"player_id": pid, "name": name, "G": g, "MPG": mpg})

    df = pd.DataFrame(rows)
    # Traded players have multiple rows; the season-total row has the highest G.
    df = df.sort_values(["player_id", "G"], ascending=[True, False])
    df = df.drop_duplicates(subset="player_id", keep="first")
    df = df[(df["MPG"] >= MIN_MPG) & (df["G"] >= MIN_GAMES_SEASON)].reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    return df.to_dict("records")


def mp_to_minutes(s: str) -> float:
    """Convert 'MM:SS' to fractional minutes."""
    m, sec = s.split(":")
    return int(m) + int(sec) / 60.0


def fetch_gamelog(pid: str, season: int) -> pd.DataFrame:
    """Fetch one player-season game log; cached as CSV."""
    cache_file = CACHE / f"{pid}_{season}.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file, parse_dates=["Date"])

    url = f"https://www.basketball-reference.com/players/{pid[0]}/{pid}/gamelog/{season}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    time.sleep(SLEEP_SEC)
    if r.status_code != 200:
        return pd.DataFrame()

    try:
        raw = pd.read_html(io.StringIO(r.text), attrs={"id": "player_game_log_reg"})[0]
    except (ValueError, IndexError):
        return pd.DataFrame()

    # Keep only rows where the player actually played: MP looks like "MM:SS".
    played = raw[raw["MP"].astype(str).str.contains(":", na=False)].copy()
    played["MP"] = played["MP"].astype(str).map(mp_to_minutes)
    played["GmSc"] = pd.to_numeric(played["GmSc"], errors="coerce")
    played["Date"] = pd.to_datetime(played["Date"], errors="coerce")
    out = played.dropna(subset=["Date", "MP", "GmSc"])[["Date", "MP", "GmSc"]].copy()
    out["player_id"] = pid
    out["season"] = season
    out.to_csv(cache_file, index=False)
    return out


def load_all() -> pd.DataFrame:
    parts = []
    for season in SEASONS:
        qualifiers = discover_qualifying_players(season)
        print(f"\n  Season {season}: {len(qualifiers)} qualifying players "
              f"(>= {MIN_MPG} MPG, >= {MIN_GAMES_SEASON} G)")
        for q in qualifiers:
            pid, name = q["player_id"], q["name"]
            try:
                d = fetch_gamelog(pid, season)
            except Exception as e:
                print(f"    ! {name:28s} {season}: {e}")
                continue
            if d.empty:
                print(f"    - {name:28s} {season}: no data")
                continue
            d = d.copy()
            d["player_name"] = name
            parts.append(d)
            print(f"    + {name:28s} {season}: {len(d):3d} games "
                  f"(qual: {q['G']}G, {q['MPG']} MPG)")
    return pd.concat(parts, ignore_index=True)


def add_recent_minutes(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Sum of MP over the previous `window` games within the same player-season."""
    df = df.sort_values(["player_id", "season", "Date"]).reset_index(drop=True)
    col = f"MP_prev_{window}"
    df[col] = (
        df.groupby(["player_id", "season"])["MP"]
          .transform(lambda s: s.shift(1).rolling(window, min_periods=window).sum())
    )
    return df.dropna(subset=[col]).reset_index(drop=True)


def report(data: pd.DataFrame, window: int) -> None:
    x = data[f"MP_prev_{window}"].to_numpy()
    y = data["GmSc"].to_numpy()
    res = stats.linregress(x, y)
    t_stat = res.slope / res.stderr
    p_one_sided = res.pvalue / 2 if res.slope < 0 else 1 - res.pvalue / 2

    print("\n=================================================================")
    print(f" Linear Regression  :   GmSc  =  b0  +  b1 * MP_prev_{window}")
    print("=================================================================")
    print(f"  observations (n)       : {len(x):,}")
    print(f"  unique players         : {data['player_id'].nunique()}")
    print(f"  MP_prev_{window} mean (sd)    : {x.mean():.1f}  ({x.std(ddof=1):.1f})")
    print(f"  GmSc mean (sd)         : {y.mean():.2f} ({y.std(ddof=1):.2f})")
    print("-----------------------------------------------------------------")
    print(f"  b0 (intercept)         : {res.intercept:.4f}")
    print(f"  b1 (slope)             : {res.slope:.6f}   GmSc per extra minute")
    print(f"  SE(b1)                 : {res.stderr:.6f}")
    print(f"  t statistic            : {t_stat:.3f}")
    print(f"  two-sided p-value      : {res.pvalue:.4g}")
    print(f"  one-sided p-value      : {p_one_sided:.4g}     (HA: b1 < 0)")
    print(f"  R^2                    : {res.rvalue**2:.5f}")
    print(f"  Pearson r              : {res.rvalue:.4f}")
    print("-----------------------------------------------------------------")
    if p_one_sided < 0.05:
        print("  Decision: REJECT H0 at alpha = 0.05.")
        print("  Evidence that heavier recent workload is associated with")
        print("  LOWER Game Scores.")
    else:
        print("  Decision: FAIL to reject H0 at alpha = 0.05.")
        print("  No statistically significant evidence that recent workload")
        print("  lowers Game Score in this sample.")
    print("=================================================================")

    # ----- Plot ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, s=6, alpha=0.12, color="steelblue", label=f"player-games (n={len(x):,})")

    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, res.intercept + res.slope * xs, color="firebrick", lw=2.2,
            label=f"OLS fit:  GmSc = {res.intercept:.2f} + ({res.slope:.4f})·MP_prev{window}")

    # Ventile means show the underlying trend more clearly than a scatter cloud.
    bins = pd.qcut(x, 20, duplicates="drop")
    grp = pd.DataFrame({"x": x, "y": y}).groupby(bins, observed=True)
    ax.plot(grp["x"].mean(), grp["y"].mean(), "o-", color="darkorange",
            lw=2, ms=6, label="binned means (ventiles)")

    ax.set_xlabel(f"Minutes played in previous {window} games")
    ax.set_ylabel("Game Score (current game)")
    ax.set_title(
        f"Recent workload vs Game Score  "
        f"(t={t_stat:.2f},  one-sided p={p_one_sided:.3g})"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = ROOT / "workload_vs_gamescore.png"
    plt.savefig(out_path, dpi=110)
    print(f"  plot saved -> {out_path.name}")


def report_within_player(data: pd.DataFrame, window: int, min_games: int = 50) -> None:
    """
    AP-Stats-style within-player analysis.

    Step 1: For each player, fit a simple linear regression
            GmSc = b0 + b1 * MP_prev{window}  using only that player's games.
            Record their slope b1_i.
    Step 2: One-sample (one-sided) t-test on the 271 slopes
            H0: mu = 0   vs   HA: mu < 0.
    """
    col = f"MP_prev_{window}"
    records = []
    for pid, grp in data.groupby("player_id"):
        if len(grp) < min_games:
            continue
        if grp[col].std(ddof=1) == 0:
            continue
        res = stats.linregress(grp[col], grp["GmSc"])
        records.append({
            "player_id": pid,
            "player_name": grp["player_name"].iloc[0],
            "n_games": len(grp),
            "slope": res.slope,
            "r": res.rvalue,
            "p_two_sided": res.pvalue,
        })
    slopes_df = pd.DataFrame(records)
    slopes = slopes_df["slope"].to_numpy()
    rs = slopes_df["r"].to_numpy()

    # One-sample one-sided t-test:  H0: mu = 0,  HA: mu < 0
    t_res = stats.ttest_1samp(slopes, popmean=0.0, alternative="less")
    n = len(slopes)
    sd = slopes.std(ddof=1)
    se = sd / np.sqrt(n)
    mean_slope = slopes.mean()

    print("\n=================================================================")
    print(f" Within-Player Analysis  (AP-scope: per-player regression + 1-sample t-test)")
    print("=================================================================")
    print(f"  players used (>= {min_games} games)  : {n}")
    print(f"  slopes (GmSc per minute of MP_prev_{window}):")
    print(f"    mean                              : {mean_slope:+.6f}")
    print(f"    sd                                : {sd:.6f}")
    print(f"    SE of mean                        : {se:.6f}")
    print(f"    median                            : {np.median(slopes):+.6f}")
    print(f"    % of players with negative slope  : {100*(slopes < 0).mean():.1f}%")
    print(f"  per-player correlations r:")
    print(f"    mean                              : {rs.mean():+.4f}")
    print(f"    median                            : {np.median(rs):+.4f}")
    print(f"    % of players with negative r      : {100*(rs < 0).mean():.1f}%")
    print("-----------------------------------------------------------------")
    print(f"  One-sample t-test on per-player slopes")
    print(f"    H0: mu_slope = 0      HA: mu_slope < 0")
    print(f"    t statistic                       : {t_res.statistic:.3f}")
    print(f"    df                                : {n - 1}")
    print(f"    one-sided p-value                 : {t_res.pvalue:.4g}")
    print("-----------------------------------------------------------------")
    if t_res.pvalue < 0.05:
        print("  Decision: REJECT H0 at alpha = 0.05.")
        print("  Evidence that, WITHIN a player, heavier recent workload is")
        print("  associated with LOWER Game Scores.")
    else:
        print("  Decision: FAIL to reject H0 at alpha = 0.05.")
        print("  Even after removing between-player differences, no significant")
        print("  evidence that recent workload lowers Game Score.")
    print("=================================================================")

    # ---- Plot: histogram of per-player slopes -----------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(slopes, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", lw=1.5, label="slope = 0  (H0)")
    ax.axvline(mean_slope, color="firebrick", lw=2,
               label=f"mean slope = {mean_slope:+.4f}")
    ax.set_xlabel(f"Per-player slope  (GmSc per minute of MP_prev_{window})")
    ax.set_ylabel("Number of players")
    ax.set_title(
        f"Distribution of within-player slopes  "
        f"(n={n} players, t={t_res.statistic:.2f}, p={t_res.pvalue:.3g})"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = ROOT / "within_player_slopes.png"
    plt.savefig(out_path, dpi=110)
    print(f"  plot saved -> {out_path.name}")

    # Save the per-player slope table for reference / write-up.
    slopes_df.sort_values("slope").to_csv(ROOT / "per_player_slopes.csv", index=False)
    print(f"  per-player table -> per_player_slopes.csv")


def main() -> None:
    print(f"Discovering qualifying players across {SEASONS}...")
    print(f"Criteria: >= {MIN_MPG} MPG and >= {MIN_GAMES_SEASON} games in the season.")
    print("(results cached in bbref_cache/, so subsequent runs are instant)\n")

    raw = load_all()
    print(f"\nTotal player-game rows pulled: {len(raw):,}")
    print(f"Unique players: {raw['player_id'].nunique()}, "
          f"player-seasons: {raw.groupby(['player_id','season']).ngroups}")

    data = add_recent_minutes(raw, window=WINDOW)
    print(f"Rows usable after computing {WINDOW}-game rolling window: {len(data):,}")

    report(data, window=WINDOW)
    report_within_player(data, window=WINDOW)


if __name__ == "__main__":
    main()

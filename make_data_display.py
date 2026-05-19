"""
Create the AP Stats data display artifacts from the cached player-game data.

Default output:
  - creates a stratified random non-overlapping 5-game block sample
  - prints a LaTeX table from the sample to the terminal
  - displays the sampled-data scatterplot with matplotlib

The script does not re-scrape any website. It uses the CSV files already stored
in bbref_cache/ and recomputes the previous-5-games workload variable.
"""
from __future__ import annotations

import argparse
import os
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).parent
LOCAL_DEPS = ROOT / ".python_deps"
if LOCAL_DEPS.exists() and any(LOCAL_DEPS.glob(f"**/*.{sys.implementation.cache_tag}-*.so")):
    sys.path.insert(0, str(LOCAL_DEPS))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

try:
    import matplotlib

    if not sys.stdout.isatty() and "MPLBACKEND" not in os.environ:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing or incompatible Python packages. In VS Code, run these commands "
        "from /Users/tonylai/Desktop/NBA_stats first:\n\n"
        "  python3 -m venv .venv\n"
        "  .venv/bin/python -m pip install -r requirements.txt\n"
        "  .venv/bin/python make_data_display.py\n"
    ) from exc

CACHE = ROOT / "bbref_cache"
WINDOW = 5
RAW_TABLE_ROWS = 12
RANDOM_SEED = 2789
BLOCK_SIZE = 5
TARGET_N = 200
TARGET_BLOCKS = TARGET_N // BLOCK_SIZE


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fix_mojibake(text: object) -> object:
    if not isinstance(text, str):
        return text
    try:
        return text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text


def ascii_name(text: object) -> object:
    if not isinstance(text, str):
        return text
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def load_player_names() -> pd.DataFrame:
    qualifiers = []
    for path in sorted(CACHE.glob("_qualifiers_*.csv")):
        df = pd.read_csv(path)
        df["season"] = int(path.stem.split("_")[-1])
        df["name"] = df["name"].map(fix_mojibake)
        df["name"] = df["name"].map(ascii_name)
        qualifiers.append(df[["player_id", "season", "name"]])
    return pd.concat(qualifiers, ignore_index=True).drop_duplicates()


def load_game_logs() -> pd.DataFrame:
    parts = []
    for path in sorted(CACHE.glob("*.csv")):
        if path.name.startswith("_qualifiers_"):
            continue
        df = pd.read_csv(path, parse_dates=["Date"])
        if {"Date", "MP", "GmSc", "player_id", "season"}.issubset(df.columns):
            parts.append(df[["Date", "MP", "GmSc", "player_id", "season"]])

    games = pd.concat(parts, ignore_index=True)
    games["MP"] = pd.to_numeric(games["MP"], errors="coerce")
    games["GmSc"] = pd.to_numeric(games["GmSc"], errors="coerce")
    games = games.dropna(subset=["Date", "MP", "GmSc"])

    names = load_player_names()
    games = games.merge(names, on=["player_id", "season"], how="inner")
    return games


def add_recent_workload(games: pd.DataFrame) -> pd.DataFrame:
    games = games.sort_values(["player_id", "season", "Date"]).reset_index(drop=True)
    games["game_number"] = games.groupby(["player_id", "season"]).cumcount() + 1
    col = f"MP_prev_{WINDOW}"
    games[col] = (
        games.groupby(["player_id", "season"])["MP"]
        .transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=WINDOW).sum())
    )
    return games.dropna(subset=[col]).reset_index(drop=True)


def stratified_block_sample(data: pd.DataFrame) -> pd.DataFrame:
    """Sample fixed, non-overlapping 5-game blocks from player-season strata."""
    rng = np.random.default_rng(RANDOM_SEED)
    candidates = []

    for (pid, season), group in data.sort_values(["player_id", "season", "Date"]).groupby(
        ["player_id", "season"], sort=True
    ):
        group = group.reset_index(drop=True)
        n_blocks = len(group) // BLOCK_SIZE
        if n_blocks == 0:
            continue
        candidates.append({
            "player_id": pid,
            "season": season,
            "group": group,
            "n_blocks": n_blocks,
        })

    if len(candidates) < TARGET_BLOCKS:
        raise ValueError(
            f"Need {TARGET_BLOCKS} player-season strata, but only "
            f"{len(candidates)} have at least {BLOCK_SIZE} usable games."
        )

    selected_indices = rng.choice(len(candidates), size=TARGET_BLOCKS, replace=False)
    blocks = []
    for selected_number, candidate_idx in enumerate(selected_indices, start=1):
        candidate = candidates[int(candidate_idx)]
        block_index = int(rng.integers(0, candidate["n_blocks"]))
        start = block_index * BLOCK_SIZE
        stop = start + BLOCK_SIZE
        block = candidate["group"].iloc[start:stop].copy()
        block["sample_block_id"] = (
            f"block_{selected_number:02d}_"
            f"{candidate['player_id']}_{candidate['season']}_chunk_{block_index + 1:02d}"
        )
        block["block_game_number"] = range(1, BLOCK_SIZE + 1)
        blocks.append(block)

    return pd.concat(blocks, ignore_index=True).sort_values(
        ["sample_block_id", "Date"]
    ).reset_index(drop=True)


def create_full_raw_table(data: pd.DataFrame) -> pd.DataFrame:
    x_col = f"MP_prev_{WINDOW}"
    columns = [
        "sample_block_id",
        "block_game_number",
        "player_id",
        "name",
        "season",
        "game_number",
        "Date",
        "MP",
        x_col,
        "GmSc",
    ]
    table = data[[col for col in columns if col in data.columns]].copy()
    table = table.rename(
        columns={
            "name": "player_name",
            "Date": "game_date",
            "MP": "minutes",
            x_col: "previous_5_game_minutes",
            "GmSc": "game_score",
        }
    )
    table["game_date"] = table["game_date"].dt.strftime("%Y-%m-%d")
    return table.sort_values(["season", "player_id", "game_date"]).reset_index(drop=True)


def create_scatterplot(data: pd.DataFrame) -> plt.Figure:
    x_col = f"MP_prev_{WINDOW}"
    x = data[x_col].to_numpy()
    y = data["GmSc"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    fit_x = np.linspace(x.min(), x.max(), 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, s=22, alpha=0.68, color="#1f5f8b", edgecolors="white", linewidths=0.25)
    ax.plot(
        fit_x,
        intercept + slope * fit_x,
        color="#c0392b",
        linewidth=2.3,
        label=f"Least-squares line: y = {intercept:.2f} + {slope:.4f}x",
    )
    ax.set_xlabel(f"Minutes played in previous {WINDOW} games")
    ax.set_ylabel("Game Score in current game")
    ax.set_title("Sampled Recent NBA Workload vs. Current-Game Performance")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def regression_summary(data: pd.DataFrame) -> dict[str, float]:
    x_col = f"MP_prev_{WINDOW}"
    x = data[x_col].to_numpy()
    y = data["GmSc"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    residuals = y - fitted
    sse = float(np.sum(residuals ** 2))
    sxx = float(np.sum((x - x.mean()) ** 2))
    df = len(x) - 2
    mse = sse / df
    stderr = float(np.sqrt(mse / sxx))
    t_stat = float(slope / stderr)
    r = float(np.corrcoef(x, y)[0, 1])
    return {
        "n": len(x),
        "intercept": float(intercept),
        "slope": float(slope),
        "stderr": stderr,
        "t_stat": t_stat,
        "df": df,
        "r": r,
        "r_squared": r ** 2,
        "x_mean": float(x.mean()),
        "x_sd": float(x.std(ddof=1)),
        "y_mean": float(y.mean()),
        "y_sd": float(y.std(ddof=1)),
    }


def create_residual_plot(data: pd.DataFrame) -> plt.Figure:
    x_col = f"MP_prev_{WINDOW}"
    x = data[x_col].to_numpy()
    y = data["GmSc"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (intercept + slope * x)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, residuals, s=12, alpha=0.35, color="#4c78a8", linewidths=0)
    ax.axhline(0, color="#c0392b", linewidth=2)
    ax.set_xlabel(f"Minutes played in previous {WINDOW} games")
    ax.set_ylabel("Residual")
    ax.set_title("Residual Plot for Sampled Workload Regression")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def create_residual_histogram(data: pd.DataFrame) -> plt.Figure:
    x_col = f"MP_prev_{WINDOW}"
    x = data[x_col].to_numpy()
    y = data["GmSc"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (intercept + slope * x)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=35, color="#4c78a8", edgecolor="white", alpha=0.9)
    ax.axvline(0, color="#c0392b", linewidth=2)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Number of sampled player-games")
    ax.set_title("Residual Histogram for Sampled Workload Regression")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def create_raw_data_table(data: pd.DataFrame) -> str:
    x_col = f"MP_prev_{WINDOW}"
    preferred_ids = [
        "jamesle01",
        "curryst01",
        "tatumja01",
        "gilgesh01",
        "doncilu01",
        "jokicni01",
        "bookede01",
        "edwaran01",
        "butleji01",
        "brunsja01",
        "halibty01",
        "duranke01",
    ]
    sample = (
        data[data["player_id"].isin(preferred_ids)]
        .sort_values(["season", "player_id", "Date"])
        .groupby("player_id", as_index=False)
        .head(1)
        .head(RAW_TABLE_ROWS)
        .copy()
    )
    if len(sample) < RAW_TABLE_ROWS:
        sample = data.sort_values(["season", "player_id", "Date"]).head(RAW_TABLE_ROWS).copy()

    sample["Date"] = sample["Date"].dt.strftime("%Y-%m-%d")
    sample["MP"] = sample["MP"].map(lambda v: f"{v:.1f}")
    sample[x_col] = sample[x_col].map(lambda v: f"{v:.1f}")
    sample["GmSc"] = sample["GmSc"].map(lambda v: f"{v:.1f}")

    columns = [
        ("name", "Player"),
        ("season", "Season"),
        ("Date", "Date"),
        ("MP", "MP"),
        (x_col, "Prev. 5 MP"),
        ("GmSc", "Game Score"),
    ]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Sample rows from the stratified random block sample used in the regression.}",
        r"\label{tab:sampled-raw-data}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        " & ".join(label for _, label in columns) + r" \\",
        r"\hline",
    ]
    for _, row in sample.iterrows():
        values = [latex_escape(row[col]) for col, _ in columns]
        lines.append(" & ".join(values) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def save_raw_data_table(data: pd.DataFrame) -> Path:
    out = ROOT / "raw_data_table.tex"
    out.write_text(create_raw_data_table(data), encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the LaTeX raw-data table and display the workload scatterplot."
    )
    parser.add_argument(
        "--save-files",
        action="store_true",
        help=(
            "Also write sampled_workload_scatterplot.png, sampled_raw_data_table.tex, "
            "sampled_workload_table.csv, and residual diagnostics to this folder."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open/display the matplotlib figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_game_logs()
    full_data = add_recent_workload(raw)
    sampled_data = stratified_block_sample(full_data)
    sampled_table = create_full_raw_table(sampled_data)
    table = create_raw_data_table(sampled_data)
    fig = create_scatterplot(sampled_data)
    stats = regression_summary(sampled_data)

    print(f"Loaded {len(raw):,} raw player-game rows.")
    print(f"Usable population after previous-{WINDOW}-game workload calculation: {len(full_data):,}.")
    print(
        f"Stratified random block sample: {len(sampled_data):,} rows "
        f"from {sampled_data['sample_block_id'].nunique()} non-overlapping "
        f"{BLOCK_SIZE}-game blocks "
        f"({len(sampled_data) / len(full_data):.2%} of usable population)."
    )
    print(
        f"Regression on sample: GmSc = {stats['intercept']:.4f} "
        f"+ ({stats['slope']:.6f}) * MP_prev_{WINDOW}; "
        f"t = {stats['t_stat']:.3f}, R^2 = {stats['r_squared']:.4f}."
    )
    print("\nLaTeX table from sampled data:\n")
    print(table)

    if args.save_files:
        plot_path = ROOT / "sampled_workload_scatterplot.png"
        table_path = ROOT / "sampled_raw_data_table.tex"
        full_table_path = ROOT / "sampled_workload_table.csv"
        residual_plot_path = ROOT / "sampled_residual_plot.png"
        residual_hist_path = ROOT / "sampled_residual_histogram.png"
        fig.savefig(plot_path, dpi=140)
        table_path.write_text(table, encoding="utf-8")
        sampled_table.to_csv(full_table_path, index=False)
        create_residual_plot(sampled_data).savefig(residual_plot_path, dpi=140)
        plt.close("all")
        create_residual_histogram(sampled_data).savefig(residual_hist_path, dpi=140)
        plt.close("all")
        print(f"Scatterplot saved to {plot_path.name}.")
        print(f"LaTeX sampled-data table saved to {table_path.name}.")
        print(f"Full sampled workload table saved to {full_table_path.name}.")
        print(f"Residual plot saved to {residual_plot_path.name}.")
        print(f"Residual histogram saved to {residual_hist_path.name}.")

    if args.no_show:
        plt.close(fig)
    else:
        print("Displaying scatterplot. Close the plot window to finish the script.")
        plt.show()


if __name__ == "__main__":
    main()

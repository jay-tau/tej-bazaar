# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "duckdb>=1.0",
#     "huggingface-hub>=0.24",
#     "polars>=1.0",
#     "pyarrow>=16.0",
# ]
# ///
"""Polars vs DuckDB on the published tejhq/indian-markets parquet.

Re-runs the six queries from the blog post "Polars vs DuckDB on 2.6M
Indian-market rows" (tejhq.dev/blog, Feb 2026) against the current layout,
one parquet per exchange per year. Both engines change speed from release to
release, so every run prints the library versions and the dataset revision
next to its timings, and checks that the two engines return the same rows.

Why this is a standalone script and not part of `pipeline/`:
- DuckDB is not a pipeline dependency. `uv run` reads the block above and
  builds a separate environment, so nothing here touches pyproject.toml.
- `uv run --with polars==X --with duckdb==Y` overrides either version, which
  is the whole point: same data, same queries, different releases.

Usage:
    uv run benchmarks/polars_vs_duckdb.py
    uv run benchmarks/polars_vs_duckdb.py --remote --json bench.jsonl
    uv run --with polars==1.30.0 --with duckdb==1.3.2 \\
        benchmarks/polars_vs_duckdb.py --revision <sha from an earlier run>
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path

import duckdb
import polars as pl
import pyarrow
from polars.testing import assert_frame_equal

REPO = "tejhq/indian-markets"
# Yearly rollups only. While the nightly compact runs, a year directory can
# briefly hold dailies too, with a different schema; this pattern skips them.
PRICES = "{ex}/year=*/{ex}_*.parquet"
UNIVERSE = "universe/nse_liquid.parquet"

SINCE = date(2024, 2, 1)  # the post's "last 24 months" cutoff, kept so runs stay comparable
ONE_DAY = date(2025, 4, 30)
YEAR_START, YEAR_END = date(2025, 1, 1), date(2025, 12, 31)


@dataclass(frozen=True)
class Sources:
    nse: list[str]
    bse: list[str]
    universe: str
    remote_nse: str  # hf:// glob, pinned to the same revision when it is known


@dataclass(frozen=True)
class Query:
    name: str
    title: str
    polars: Callable[[Sources], pl.LazyFrame]
    duckdb: Callable[[Sources], str]
    remote: bool = False


# `year` and `month` are stored columns as well as path segments, so hive
# partitioning is off in both engines rather than left to each one's default.
def scan(paths: str | list[str]) -> pl.LazyFrame:
    return pl.scan_parquet(paths, hive_partitioning=False)


def read(paths: str | list[str]) -> str:
    if isinstance(paths, str):
        paths = [paths]
    quoted = ", ".join("'" + p.replace("'", "''") + "'" for p in paths)
    return f"read_parquet([{quoted}], hive_partitioning = false)"


def reliance(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.filter((pl.col("symbol") == "RELIANCE") & (pl.col("date") >= SINCE))


# The post joined a ~1.5K-row sectors table that was never published. The 2025
# top-100 slice of the point-in-time universe is the same shape (1,200 rows)
# and is the join the dataset card documents.
def liquid100_polars(s: Sources) -> pl.LazyFrame:
    in_2025 = pl.col("rebalance_date").is_between(YEAR_START, YEAR_END)
    members = (
        scan(s.universe)
        .filter((pl.col("rank") <= 100) & in_2025)
        .select("isin", "rebalance_date", "valid_to", "rank")
    )
    return (
        scan(s.nse)
        .filter(pl.col("date").is_between(YEAR_START, YEAR_END))
        .join(members, on="isin")
        .filter(pl.col("date").is_between(pl.col("rebalance_date"), pl.col("valid_to")))
        .select("date", "symbol", "close", "rank")
    )


def liquid100_duckdb(s: Sources) -> str:
    return f"""
        SELECT p.date, p.symbol, p.close, u.rank
        FROM {read(s.nse)} p
        JOIN {read(s.universe)} u
          ON u.isin = p.isin AND p.date BETWEEN u.rebalance_date AND u.valid_to
        WHERE u.rank <= 100
          AND u.rebalance_date BETWEEN DATE '{YEAR_START}' AND DATE '{YEAR_END}'
          AND p.date BETWEEN DATE '{YEAR_START}' AND DATE '{YEAR_END}'
    """


def sma20_polars(s: Sources) -> pl.LazyFrame:
    return (
        scan(s.nse)
        .filter(pl.col("date") >= YEAR_START)
        .select("symbol", "date", "close")
        .sort("symbol", "date")
        .with_columns(pl.col("close").rolling_mean(20).over("symbol").alias("sma20"))
    )


def sma20_duckdb(s: Sources) -> str:
    # The CASE matches Polars' rolling_mean, which leaves the first 19 rows of
    # each symbol null instead of averaging a partial window.
    return f"""
        SELECT symbol, date, close,
               CASE WHEN count(close) OVER w = 20 THEN avg(close) OVER w END AS sma20
        FROM {read(s.nse)}
        WHERE date >= DATE '{YEAR_START}'
        WINDOW w AS (PARTITION BY symbol ORDER BY date
                     ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
    """


QUERIES = [
    Query(
        "Q1",
        "full scan, count by series",
        lambda s: scan(s.nse + s.bse).group_by("series").agg(pl.len().alias("n")),
        lambda s: f"SELECT series, count(*) AS n FROM {read(s.nse + s.bse)} GROUP BY series",
    ),
    Query(
        "Q2",
        f"RELIANCE since {SINCE}",
        lambda s: reliance(scan(s.nse)),
        lambda s: (
            f"SELECT * FROM {read(s.nse)} WHERE symbol = 'RELIANCE' AND date >= DATE '{SINCE}'"
        ),
    ),
    Query(
        "Q3",
        f"RELIANCE since {SINCE}, over hf://",
        lambda s: reliance(scan(s.remote_nse)).select("date", "close"),
        lambda s: (
            f"SELECT date, close FROM {read(s.remote_nse)} "
            f"WHERE symbol = 'RELIANCE' AND date >= DATE '{SINCE}'"
        ),
        remote=True,
    ),
    Query(
        "Q4",
        f"all NSE symbols on {ONE_DAY}",
        lambda s: scan(s.nse).filter(pl.col("date") == ONE_DAY),
        lambda s: f"SELECT * FROM {read(s.nse)} WHERE date = DATE '{ONE_DAY}'",
    ),
    Query("Q5", "2025 closes joined to the liquid-100", liquid100_polars, liquid100_duckdb),
    Query("Q6", f"20-day SMA since {YEAR_START}", sma20_polars, sma20_duckdb),
]


def arrow_table(rel: duckdb.DuckDBPyRelation) -> pyarrow.Table:
    # DuckDB 1.5 renamed fetch_arrow_table() to to_arrow_table() and
    # deprecated the old name; older releases only have the old one.
    fetch = getattr(rel, "to_arrow_table", None) or rel.fetch_arrow_table
    return fetch()


def run_polars(query: Query, src: Sources) -> tuple[pl.DataFrame, float]:
    start = time.perf_counter()
    df = query.polars(src).collect()
    return df, time.perf_counter() - start


def run_duckdb(query: Query, src: Sources) -> tuple[pl.DataFrame, float]:
    # A new in-memory database per run, so DuckDB cannot answer the remote
    # query from its in-process file cache while Polars refetches every time.
    con = duckdb.connect()
    try:
        start = time.perf_counter()
        table = arrow_table(con.sql(query.duckdb(src)))
        elapsed = time.perf_counter() - start
    finally:
        con.close()
    return pl.from_arrow(table), elapsed


def measure(
    once: Callable[[], tuple[pl.DataFrame, float]], runs: int
) -> tuple[pl.DataFrame, list[float]]:
    result, _ = once()  # warm-up; its result is the one compared across engines
    return result, [once()[1] for _ in range(runs)]


def same_rows(a: pl.DataFrame, b: pl.DataFrame) -> bool:
    if a.columns != b.columns:
        return False
    try:
        # Row order is not part of any query, and the count dtypes differ
        # (UInt32 vs Int64), so compare sorted values only.
        assert_frame_equal(a.sort(a.columns), b.sort(b.columns), check_dtypes=False)
    except AssertionError:
        return False
    return True


def cpu_model() -> str:
    if sys.platform == "darwin":
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        )
        if out.stdout.strip():
            return out.stdout.strip()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def environment() -> dict:
    with duckdb.connect() as con:
        # fetchall(), not fetchone(): DuckDB 2.0 dropped fetchone() from relations.
        duckdb_threads = con.sql("SELECT current_setting('threads')").fetchall()[0][0]
    return {
        "python": platform.python_version(),
        "polars": pl.__version__,
        "duckdb": duckdb.__version__,
        "pyarrow": pyarrow.__version__,
        "os": f"{platform.system()} {platform.release()}",
        "cpu": cpu_model(),
        "cores": os.cpu_count(),
        "polars_threads": pl.thread_pool_size(),
        "duckdb_threads": duckdb_threads,
    }


def snapshot(revision: str) -> tuple[Path, str]:
    """Download (or reuse the cached) rollups. Returns (root, commit sha)."""
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            REPO,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[PRICES.format(ex="nse"), PRICES.format(ex="bse"), UNIVERSE],
        )
    )
    # The cache lays snapshots out as snapshots/<commit sha>, even for "main".
    return root, root.name


def seconds(t: float) -> str:
    return f"{t:.3f}s" if t < 10 else f"{t:.1f}s"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", type=int, default=5, help="timed runs per query (default 5)")
    p.add_argument(
        "--revision",
        default="main",
        help="dataset branch or commit; pin a sha to compare versions on identical data",
    )
    p.add_argument(
        "--data",
        type=Path,
        help="local tree with the same layout (e.g. data/out) instead of the HF snapshot",
    )
    p.add_argument(
        "--remote", action="store_true", help="also run Q3, which reads over hf:// (slow)"
    )
    p.add_argument("--json", type=Path, help="append this run's results as one JSON line")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.data:
        root, revision = args.data, args.revision
        label = f"{args.data} (local)"
    else:
        root, revision = snapshot(args.revision)
        label = f"{REPO} @ {revision}"

    src = Sources(
        nse=sorted(str(p) for p in root.glob(PRICES.format(ex="nse"))),
        bse=sorted(str(p) for p in root.glob(PRICES.format(ex="bse"))),
        universe=str(root / UNIVERSE),
        remote_nse=f"hf://datasets/{REPO}@{revision}/{PRICES.format(ex='nse')}",
    )
    if not src.nse or not Path(src.universe).exists():
        raise SystemExit(f"no NSE rollups or {UNIVERSE} under {root}")
    files = len(src.nse) + len(src.bse)
    total = scan(src.nse + src.bse).select(pl.len()).collect().item()

    env = environment()
    print(f"{label}: {files} files, {total:,} rows")
    print(
        f"polars {env['polars']}, duckdb {env['duckdb']}, pyarrow {env['pyarrow']}, "
        f"Python {env['python']}"
    )
    print(
        f"{env['os']}, {env['cpu']}, {env['cores']} cores "
        f"(threads: polars {env['polars_threads']}, duckdb {env['duckdb_threads']})"
    )
    print(f"median of {args.runs} runs after one warm-up\n")
    print(f"{'query':<44}{'rows':>9}{'polars':>10}{'duckdb':>10}  {'faster':<14}match")

    results, mismatched = [], []
    for q in QUERIES:
        if q.remote and not args.remote:
            continue
        p_df, p_times = measure(partial(run_polars, q, src), args.runs)
        d_df, d_times = measure(partial(run_duckdb, q, src), args.runs)
        p_med, d_med = statistics.median(p_times), statistics.median(d_times)
        match = same_rows(p_df, d_df)
        if not match:
            mismatched.append(f"{q.name}: polars {p_df.shape}, duckdb {d_df.shape}")
        winner = "polars" if p_med <= d_med else "duckdb"
        ratio = max(p_med, d_med) / min(p_med, d_med)
        print(
            f"{q.name + ' ' + q.title:<44}{p_df.height:>9,}{seconds(p_med):>10}"
            f"{seconds(d_med):>10}  {f'{winner} {ratio:.1f}x':<14}{'yes' if match else 'NO'}"
        )
        for engine, df, times in (("polars", p_df, p_times), ("duckdb", d_df, d_times)):
            results.append(
                {
                    "query": q.name,
                    "engine": engine,
                    "rows": df.height,
                    "median_s": statistics.median(times),
                    "times_s": times,
                    "match": match,
                }
            )

    for line in mismatched:
        print(f"\nresults differ, {line}")

    if args.json:
        record = {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "env": env,
            "dataset": {"source": label, "revision": revision, "files": files, "rows": total},
            "runs": args.runs,
            "results": results,
        }
        with args.json.open("a") as f:
            f.write(json.dumps(record) + "\n")
    return 1 if mismatched else 0


if __name__ == "__main__":
    raise SystemExit(main())

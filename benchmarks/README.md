# Benchmarks

`polars_vs_duckdb.py` re-runs the six queries from the blog post
[Polars vs DuckDB on 2.6M Indian-market rows](https://tejhq.dev/blog/polars-vs-duckdb-india-eod-benchmark)
against the published dataset. Both libraries release every few weeks and
their relative speed moves with them, so every run prints the library
versions, machine and dataset revision next to its timings.

## Run

```bash
uv run benchmarks/polars_vs_duckdb.py            # local queries, latest releases
uv run benchmarks/polars_vs_duckdb.py --remote   # adds Q3, read over hf://
```

`uv run` builds the script's own environment from its inline dependency
block, so DuckDB never becomes a pipeline dependency. uv keeps that
environment between runs, so later runs stay on the releases it first
installed; add `--upgrade` to move to the newest ones. The first run downloads
the yearly NSE and BSE rollups and the universe (about 230 MB) into the
Hugging Face cache; later runs reuse it. `--data data/out` runs the same
queries against a tree you built with the pipeline.

Each query runs once to warm up, then `--runs` times (default 5), and the
table shows the median. Each DuckDB run opens a new in-memory database, so
neither engine carries a cache from one run to the next beyond the OS page
cache. Both engines must return the same rows: a mismatch is printed and the
script exits 1.

## Compare releases

Pin the dataset revision so both runs read the same rows (each run prints its
sha), and override versions with `--with`:

```bash
uv run --upgrade benchmarks/polars_vs_duckdb.py --revision <sha> --json bench.jsonl
uv run --python 3.12 --with polars==1.0.0 --with duckdb==1.0.0 \
    benchmarks/polars_vs_duckdb.py --revision <sha> --json bench.jsonl
```

Older releases may have no wheels for a new Python; `--python 3.12` picks an
interpreter they support. `--json` appends one line per run with the
versions, machine, revision and every timing.

## Queries

| | Query | Change from the post |
|---|---|---|
| Q1 | Rows per `series`, NSE and BSE, all years | none |
| Q2 | RELIANCE since 2024-02-01 | none |
| Q3 | Q2 over `hf://`, `date` and `close` only, pinned to the same revision (`--remote`) | none |
| Q4 | All NSE rows on 2025-04-30 | the post didn't name the day |
| Q5 | 2025 NSE closes joined on ISIN to the liquid-100 in force on each date | the post's ~1.5K-row sectors table was never published; the 2025 top-100 slice of `universe/nse_liquid.parquet` is 1,200 rows |
| Q6 | 20-day moving average of close per symbol since 2025-01-01 | the first 19 rows of each symbol are null in both engines, as with Polars' `rolling_mean`; the post's SQL averaged the partial windows |

The post read one small file per trading day. The dataset now has one file
per exchange per year (see
[One parquet per year, and the glob that ate DuckDB](https://tejhq.dev/blog/one-parquet-per-year)),
so its timings are not comparable with these. `year` and `month` are stored
columns in those files, so hive partitioning is off in both engines.

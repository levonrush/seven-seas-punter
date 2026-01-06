# Data

This folder stores local data inputs and intermediate outputs.
It is intentionally ignored by git.

Common files:
- `data/data.tar`: Betfair historic archive (download target).
- `data/db.duckdb`: local database built from ingest.
- `data/features_cutoff_*.parquet` / `data/features_cutoff_*.csv`: built feature sets.
- `data/historic_manifest.json`: historic download manifest (resume support).
- `data/ingest_manifest.json`: ingest manifest (incremental ingest).
- `data/historic_lists/`: cached file lists from the historic API.

## Download historic data

If you do not already have `data/data.tar`, download it first:

```
punter download-historic --auto --download --output data/data.tar
```

Notes:
- `--auto` walks the available months and pulls whatever is purchasable.
- `--download` writes into `data/data.tar` and updates `data/historic_manifest.json`.
- Use `--workers 1` if you want to be gentle on the API.
- Use `--force` to rebuild the tar from scratch.

## Ingest into DuckDB

Once `data/data.tar` is present:

```
punter ingest --archive data/data.tar
```

To ingest only new files added since last time:

```
punter ingest --archive data/data.tar --ingest-new
```

Avoid `--force-ingest` unless you are writing to a fresh database, because
it will reprocess the entire archive and can duplicate snapshot rows.

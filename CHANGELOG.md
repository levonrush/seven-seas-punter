# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- Added automatic DuckDB schema recovery with backup on failure.
- Added elapsed-time logging across workflow steps and shared modules.
- Defaulted ingest filters to AU horse racing WIN markets to reduce data volume.
- Made ingest filtering optional (default off), with a `--filter-au-win` flag.
- Added logo asset and playful Punters Club intro to README.
- Improved elapsed-time logs to show seconds/minutes/hours as needed.
- Added ingest short-circuit when snapshots already exist, with `--force-ingest` to override.

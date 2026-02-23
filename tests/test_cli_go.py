import argparse
import sys
import types

# `workflow.cli` imports storage modules that import duckdb. For CLI wiring tests we only
# need parser/dispatch behavior, so we stub duckdb when the dependency is unavailable.
sys.modules.setdefault(
    "duckdb",
    types.SimpleNamespace(DuckDBPyConnection=object, connect=lambda *args, **kwargs: None),
)

from shared.utils.cli_presets import go_historic_args, go_pipeline_args, recommended_historic_workers
from workflow.cli import build_parser, cmd_go, cmd_repair_manifests


def test_recommended_historic_workers_is_bounded():
    assert recommended_historic_workers(cpu_count=1) == 2
    assert recommended_historic_workers(cpu_count=4) == 4
    assert recommended_historic_workers(cpu_count=64) == 8


def test_cmd_go_runs_historic_then_pipeline(monkeypatch):
    parsed_tokens = []
    executed = []

    class _FakeParser:
        """Capture parsed token lists and run synthetic command handlers."""

        def parse_args(self, tokens):  # noqa: ANN001 - mirror argparse signature
            parsed_tokens.append(list(tokens))
            command = tokens[0]

            def _runner(_args):  # noqa: ANN001 - mirror argparse callback signature
                executed.append(command)

            return argparse.Namespace(func=_runner)

    monkeypatch.setattr("workflow.cli.build_parser", lambda: _FakeParser())

    cmd_go(argparse.Namespace())

    assert executed == ["download-historic", "pipeline"]
    assert parsed_tokens[0][0] == "download-historic"
    assert parsed_tokens[0][1:] == go_historic_args()
    assert parsed_tokens[1][0] == "pipeline"
    assert parsed_tokens[1][1:] == go_pipeline_args()


def test_parser_accepts_repair_manifests_command():
    parser = build_parser()
    args = parser.parse_args(["repair-manifests", "--no-backup"])

    assert args.command == "repair-manifests"
    assert args.func is cmd_repair_manifests
    assert args.bad_list == "artifacts/ingest_bad_stream_members.txt"
    assert args.historic_manifest == "data/historic_manifest.json"
    assert args.ingest_manifest == "data/ingest_manifest.json"
    assert args.backup is False


def test_cmd_repair_manifests_handles_missing_files(monkeypatch):
    def fake_repair_manifests(**kwargs):  # noqa: ANN001 - mirror runtime signature
        raise FileNotFoundError("missing file")

    monkeypatch.setattr("workflow.cli.repair_manifests", fake_repair_manifests)

    cmd_repair_manifests(
        argparse.Namespace(
            bad_list="missing.txt",
            historic_manifest="historic.json",
            ingest_manifest="ingest.json",
            backup=True,
        )
    )


def test_parser_backtest_defaults_to_win_market_type():
    parser = build_parser()
    args = parser.parse_args(["backtest"])
    assert args.market_types == "WIN"


def test_parser_pipeline_has_decision_market_type_default():
    parser = build_parser()
    args = parser.parse_args(["pipeline"])
    assert args.market_types == "ALL"
    assert args.decision_market_types == "WIN"

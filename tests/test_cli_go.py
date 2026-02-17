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
from workflow.cli import cmd_go


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


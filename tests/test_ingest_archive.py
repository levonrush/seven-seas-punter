import bz2
import io
import sys
import tarfile
import types

sys.modules.setdefault(
    "duckdb",
    types.SimpleNamespace(DuckDBPyConnection=object, connect=lambda *args, **kwargs: None),
)

from workflow.ingest_archive import _ingest_bz_stream_archive


class _DummyStore:
    """No-op storage stand-in so stream ingest behavior can be tested without DuckDB."""

    def upsert_markets(self, rows):  # noqa: ANN001 - mirrors real store API
        """Accept market rows without writing to a database."""
        return None

    def upsert_runners(self, rows):  # noqa: ANN001 - mirrors real store API
        """Accept runner rows without writing to a database."""
        return None

    def upsert_results(self, rows):  # noqa: ANN001 - mirrors real store API
        """Accept result rows without writing to a database."""
        return None

    def append_snapshots(self, rows):  # noqa: ANN001 - mirrors real store API
        """Accept snapshot rows without writing to a database."""
        return None


def _add_tar_member(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    """Add a byte payload to a tar archive using a stable member definition."""
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def test_ingest_manifest_skips_corrupt_members(tmp_path):
    archive_path = tmp_path / "sample.tar"
    good_payload = bz2.compress(b'{"pt":1,"mc":[]}\n')
    bad_payload = b"<!DOCTYPE html><title>package_not_purchased</title>"
    with tarfile.open(archive_path, "w") as tar:
        _add_tar_member(tar, "good.bz2", good_payload)
        _add_tar_member(tar, "bad.bz2", bad_payload)

    manifest = {"members": {}, "_members": set(), "version": 1}
    bad_members_output = tmp_path / "bad_members.txt"
    counts = _ingest_bz_stream_archive(
        archive_path=archive_path,
        store=_DummyStore(),
        workers=1,
        manifest=manifest,
        progress_every=0,
        bad_members_output=bad_members_output,
    )

    assert counts["bad_files"] == 1
    assert "good.bz2" in manifest["members"]
    assert "bad.bz2" not in manifest["members"]
    assert bad_members_output.exists()
    assert bad_members_output.read_text(encoding="utf-8").strip() == "bad.bz2"

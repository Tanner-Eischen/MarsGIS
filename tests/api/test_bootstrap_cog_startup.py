"""API tests for COG bootstrap startup."""

from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

from marshab.core.datasets.bootstrap_cog import (
    MIN_EXPECTED_SIZE_BYTES,
    ensure_mola_cog_present,
)

FIXTURE_COG = Path(__file__).resolve().parent.parent / "fixtures" / "dem_small.cog.tif"


def _make_mock_response(size_bytes: int) -> BytesIO:
    """Create BytesIO with given size to simulate HTTP response body."""
    return BytesIO(b"x" * size_bytes)


class TestBootstrapCog:
    """Tests for ensure_mola_cog_present."""

    def test_skips_download_when_file_exists(self, monkeypatch):
        """Skips download when COG path exists."""
        monkeypatch.setenv("MARSHAB_MOLA_COG_PATH", str(FIXTURE_COG))
        monkeypatch.setenv("MARSHAB_MOLA_COG_URL", "https://example.com/fake.cog.tif")
        ensure_mola_cog_present()
        assert FIXTURE_COG.exists()

    def test_fails_fast_when_url_missing_and_file_missing(self, monkeypatch, tmp_path):
        """Fails fast when COG missing and MARSHAB_MOLA_COG_URL not set."""
        missing_path = tmp_path / "nonexistent.cog.tif"
        assert not missing_path.exists()
        monkeypatch.setenv("MARSHAB_MOLA_COG_PATH", str(missing_path))
        monkeypatch.setenv("MARSHAB_MOLA_COG_URL", "")
        monkeypatch.delenv("MARSHAB_MOLA_COG_SHA256", raising=False)

        with pytest.raises(RuntimeError) as exc_info:
            ensure_mola_cog_present()
        assert "MARSHAB_MOLA_COG_URL" in str(exc_info.value)

    def test_downloads_when_missing(self, monkeypatch, tmp_path):
        """Downloads when file missing and URL set."""
        dest = tmp_path / "test.cog.tif"
        assert not dest.exists()
        monkeypatch.setenv("MARSHAB_MOLA_COG_PATH", str(dest))
        monkeypatch.setenv("MARSHAB_MOLA_COG_URL", "https://example.com/fake.cog.tif")
        monkeypatch.delenv("MARSHAB_MOLA_COG_SHA256", raising=False)

        mock_body = _make_mock_response(MIN_EXPECTED_SIZE_BYTES + 1024)

        def mock_urlopen(req, timeout=60):
            body = _make_mock_response(MIN_EXPECTED_SIZE_BYTES + 1024)

            class MockResp:
                def read(self, size=-1):
                    return body.read(size) if size > 0 else body.read()

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

            return MockResp()

        with patch("marshab.core.datasets.bootstrap_cog.urllib.request.urlopen", mock_urlopen):
            ensure_mola_cog_present()

        assert dest.exists()
        assert dest.stat().st_size >= MIN_EXPECTED_SIZE_BYTES
        assert not dest.with_suffix(dest.suffix + ".partial").exists()

    def test_verifies_sha256_when_configured(self, monkeypatch, tmp_path):
        """Verifies SHA256 when MARSHAB_MOLA_COG_SHA256 is set."""
        dest = tmp_path / "test.cog.tif"
        monkeypatch.setenv("MARSHAB_MOLA_COG_PATH", str(dest))
        monkeypatch.setenv("MARSHAB_MOLA_COG_URL", "https://example.com/fake.cog.tif")

        # Create known content and compute its hash
        known_content = b"x" * (MIN_EXPECTED_SIZE_BYTES + 1024)
        import hashlib

        expected_hash = hashlib.sha256(known_content).hexdigest().lower()
        monkeypatch.setenv("MARSHAB_MOLA_COG_SHA256", expected_hash)

        mock_body = BytesIO(known_content)

        def mock_urlopen(req, timeout=60):
            class MockResp:
                def read(self, size=-1):
                    return mock_body.read(size)

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

            return MockResp()

        with patch("marshab.core.datasets.bootstrap_cog.urllib.request.urlopen", mock_urlopen):
            ensure_mola_cog_present()

        assert dest.exists()
        assert not dest.with_suffix(dest.suffix + ".partial").exists()

    def test_skips_when_path_not_set(self, monkeypatch):
        """Skips bootstrap when MARSHAB_MOLA_COG_PATH not set."""
        monkeypatch.delenv("MARSHAB_MOLA_COG_PATH", raising=False)
        monkeypatch.delenv("MARSHAB_MOLA_COG_URL", raising=False)
        ensure_mola_cog_present()  # Should not raise

"""Tests for tap."""

import tap


def test_version() -> None:
    """Package version is accessible."""
    assert isinstance(tap.__version__, str)
    parts = tap.__version__.split(".")
    assert len(parts) >= 2

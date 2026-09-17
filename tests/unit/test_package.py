"""Smoke test proving the package installs and imports in a fresh environment."""

import bio_research_copilot


def test_package_exposes_version() -> None:
    assert bio_research_copilot.__version__ == "0.1.0"

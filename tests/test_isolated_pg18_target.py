from pathlib import Path

import pytest

from app.db.isolated_test_target import (
    validate_isolated_pg18_configuration,
    validate_isolated_pg18_target,
)


def _case(tmp_path: Path):
    directory = tmp_path / ".local-tools" / "isolated_pg18"
    directory.mkdir(parents=True)
    return {
        "async_url": "postgresql+asyncpg://tester:secret@127.0.0.1:55567/assistant",
        "sync_url": "postgresql://tester:secret@127.0.0.1:55567/assistant",
        "expected_directory": str(directory),
        "actual_directory": str(directory),
        "actual_database": "assistant",
        "actual_port": "55567",
        "server_version": "18.6",
        "repository_root": tmp_path,
    }


def test_isolated_pg18_target_accepts_matching_local_cluster(tmp_path):
    validate_isolated_pg18_target(**_case(tmp_path))


@pytest.mark.parametrize("change", [
    {"async_url": "postgresql+asyncpg://tester:secret@127.0.0.1:5432/assistant"},
    {"async_url": "postgresql+asyncpg://tester:secret@127.0.0.1:55432/assistant"},
    {"async_url": "postgresql+asyncpg://tester:secret@127.0.0.1:55567/personal_assistant"},
    {"sync_url": "postgresql://tester:secret@127.0.0.1:5432/assistant"},
    {"actual_database": "personal_assistant"},
    {"actual_port": "5432"},
    {"server_version": "16.9"},
])
def test_isolated_pg18_target_rejects_production_or_mismatch(tmp_path, change):
    case = _case(tmp_path)
    case.update(change)
    with pytest.raises(ValueError):
        validate_isolated_pg18_target(**case)


def test_isolated_pg18_configuration_rejects_directory_outside_local_tools(tmp_path):
    case = _case(tmp_path)
    outside = tmp_path / "production"
    outside.mkdir()
    with pytest.raises(ValueError):
        validate_isolated_pg18_configuration(
            case["async_url"], case["sync_url"], str(outside),
            repository_root=tmp_path,
        )

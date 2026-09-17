import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))


@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch):
	db_path = tmp_path / "progresssync.db"
	monkeypatch.setenv("PROGRESSSYNC_DB", str(db_path))
	import backend.app as module
	module.DB_PATH = db_path
	module.init_db()
	yield

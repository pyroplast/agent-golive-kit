import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def kb_dir():
    return FIXTURES / "kb"


@pytest.fixture
def config():
    return json.loads((ROOT / "config" / "client.json").read_text())


@pytest.fixture
def chunks(kb_dir):
    from deploykit.kb import load_kb
    return load_kb(kb_dir)


@pytest.fixture
def scenarios():
    return json.loads((FIXTURES / "scenarios.json").read_text())

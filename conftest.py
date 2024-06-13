import pytest
import sys as sys


@pytest.fixture  # mcoding https://www.youtube.com/watch?v=DhUpxWjOhME 16:00
def capture_stdout(monkeypatch):
    buffer = {"stdout": "", "write_calls": 0}

    def fake_write(s):
        buffer["stdout"] += s
        buffer["write_calls"] += 1

    monkeypatch.setattr(sys.stdout, "write", fake_write)
    return buffer

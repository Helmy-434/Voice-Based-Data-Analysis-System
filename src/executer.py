import ast
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

TIMEOUT = 15
MAX_OUTPUT = 20_000


def run_code(code: str, timeout: int = TIMEOUT) -> dict:

    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "snippet.py"
        script.write_text(code, encoding="utf-8")
        try:
            p = subprocess.run(
                [sys.executable, str(script)],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
                stdin=subprocess.DEVNULL,
                encoding="utf-8",
                errors="replace",
            )
            return {
                "stdout": p.stdout[:MAX_OUTPUT],
                "stderr": p.stderr[:MAX_OUTPUT],
                "returncode": p.returncode,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as e:
            stdout_str = e.stdout or ""
            if isinstance(stdout_str, bytes):
                stdout_str = stdout_str.decode(errors="replace")
            return {
                "stdout": stdout_str[:MAX_OUTPUT],
                "stderr": f"Execution timed out after {timeout}s.",
                "returncode": -1,
                "timed_out": True,
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "timed_out": False,
            }
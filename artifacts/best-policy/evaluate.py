"""Compatibility entry point; prefer scripts/run.py evaluate."""
from pathlib import Path
import subprocess
import sys
root = Path(__file__).resolve().parents[2]
subprocess.run([sys.executable, str(root / 'scripts/run.py'), 'evaluate', *sys.argv[1:]], check=True)

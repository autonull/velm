import sys
from pathlib import Path

# Ensure the src/ directory is on the import path so `velm.*` resolves
_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

"""Make ``src/`` importable without an editable install.

Inserted at collection time so ``pytest`` works whether or not the package
has been ``pip install -e``'d.
"""

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

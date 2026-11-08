import sys
from pathlib import Path

# Ensure workspace root is always in sys.path so ai_workers and root modules are accessible
_workspace_root = str(Path(__file__).resolve().parent.parent.parent)
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

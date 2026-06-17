"""Component smoke test for constrained machines."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.validate import run_component_checks

print(json.dumps(run_component_checks(), indent=2))

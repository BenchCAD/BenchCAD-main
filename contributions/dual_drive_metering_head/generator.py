"""Return self-contained CadQuery source for one deterministic variant."""
import importlib.util
from pathlib import Path

_path = Path(__file__).with_name("design.py")
_spec = importlib.util.spec_from_file_location("local_design", _path)
_design = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_design)

def build(difficulty, seed):
    return _design.code_for('dual_drive_metering_head', difficulty, seed)

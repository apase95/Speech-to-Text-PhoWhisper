"""Legacy command and import alias."""
import sys
from pathlib import Path

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import runpy
    runpy.run_module('scripts.data.human_reference_review', run_name='__main__')
else:
    from scripts.data import human_reference_review as implementation
    sys.modules[__name__] = implementation

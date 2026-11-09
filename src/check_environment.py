"""Legacy environment diagnostic command and import alias."""
import sys
from pathlib import Path

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import runpy
    runpy.run_module('scripts.env.check_environment', run_name='__main__')
else:
    from scripts.env import check_environment as implementation
    sys.modules[__name__] = implementation

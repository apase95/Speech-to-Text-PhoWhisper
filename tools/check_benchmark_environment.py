"""Legacy command and import alias."""
import sys
from pathlib import Path

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.env import check_benchmark_environment as implementation

if __name__ == '__main__':
    import runpy
    runpy.run_module('scripts.env.check_benchmark_environment', run_name='__main__')
else:
    sys.modules[__name__] = implementation

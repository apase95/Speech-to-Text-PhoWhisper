"""Legacy command and import alias."""
import sys
from pathlib import Path
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import runpy
    runpy.run_module('benchmarks.runners.execute_staged_fleurs60', run_name='__main__')
else:
    from benchmarks.runners import execute_staged_fleurs60 as implementation
    sys.modules[__name__] = implementation

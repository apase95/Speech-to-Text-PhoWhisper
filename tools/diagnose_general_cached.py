"""Legacy command and import alias."""
import sys
from pathlib import Path
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import runpy
    runpy.run_module('benchmarks.audits.diagnose_general_cached', run_name='__main__')
else:
    from benchmarks.audits import diagnose_general_cached as implementation
    sys.modules[__name__] = implementation

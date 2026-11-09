"""Legacy command and import alias."""
import sys
from pathlib import Path
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import runpy
    runpy.run_module('benchmarks.audits.analyze_matched_rescoring', run_name='__main__')
else:
    from benchmarks.audits import analyze_matched_rescoring as implementation
    sys.modules[__name__] = implementation

"""Legacy command and import alias."""
import sys
from pathlib import Path

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import runpy
    runpy.run_module('scripts.data.download_viet_bud500_subset', run_name='__main__')
else:
    from scripts.data import download_viet_bud500_subset as implementation
    sys.modules[__name__] = implementation

"""Legacy command and import alias."""
import sys
from pathlib import Path

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.convert import convert_phowhisper as implementation

if __name__ == '__main__':
    implementation.main()
else:
    sys.modules[__name__] = implementation

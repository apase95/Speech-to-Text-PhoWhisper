"""Compatibility alias; patches address the canonical module's globals."""
import sys

if __name__ == '__main__':
    from phowhisper.cli import main
    main()
else:
    from phowhisper import pipeline
    sys.modules[__name__] = pipeline

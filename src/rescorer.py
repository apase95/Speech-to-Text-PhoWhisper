"""Compatibility alias for phowhisper.rescorer."""
import sys
from phowhisper import rescorer
sys.modules[__name__] = rescorer

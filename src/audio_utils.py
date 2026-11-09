"""Compatibility alias for phowhisper.audio_utils."""
import sys
from phowhisper import audio_utils
sys.modules[__name__] = audio_utils

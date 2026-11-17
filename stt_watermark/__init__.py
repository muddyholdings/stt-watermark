"""stt-watermark: inaudible spread-spectrum audio watermark for AI-generated speech.

See https://stt.ai for the
algorithm, threat model, and robustness gates.
"""

from stt_watermark.embed import embed
from stt_watermark.detect import DetectResult, detect

__version__ = "0.2.0a1"
__all__ = ["embed", "detect", "DetectResult", "__version__"]

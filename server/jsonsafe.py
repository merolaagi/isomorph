"""Make results safe for JSON: NaN and infinity become null.

Browsers and FastAPI reject non-finite floats, and a single NaN deep inside a
report (for example a spectrum of an all-zero tensor) would otherwise turn
the whole response into an unexplained 500."""
import math

import numpy as np


def clean(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.generic):
        return clean(o.item())
    return o

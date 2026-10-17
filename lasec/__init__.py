"""LASeC: sentence-level drone log event abstraction via semantic clustering, for forensic triage."""
__version__ = "2.0.0-dev"

import os as _os

# joblib/loky cannot count physical cores on some Windows set-ups and prints a long warning
_os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_os.cpu_count() or 1))

"""Small gym compatibility shim for the WiFi simulation demo.

The WiFi eval environment only needs ``gym.spaces.Box`` and
``gym.spaces.Discrete``. This keeps the web demo runnable on machines where the
legacy gym package is not installed.
"""

from . import spaces

__all__ = ["spaces"]

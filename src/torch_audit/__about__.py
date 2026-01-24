"""Package metadata.

Keeping these values in a dedicated module avoids importing the full library
at package import time (useful for lightweight CLIs and tooling).
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

__title__ = "torch-audit"
__description__ = "The Linter for PyTorch: Detects silent training bugs."
__url__ = "https://github.com/RMalkiv/torch-audit"
__author__ = "Roman Malkiv"
__author_email__ = "malkiv.roman@gmail.com"
__license__ = "MIT"

try:
    __version__ = _version(__title__)
except PackageNotFoundError:  # pragma: no cover
    # Package is not installed (e.g. editable checkout without metadata)
    __version__ = "0.0.0-dev"

__all__ = [
    "__title__",
    "__description__",
    "__url__",
    "__author__",
    "__author_email__",
    "__license__",
    "__version__",
]

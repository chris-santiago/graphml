"""Package initialization."""
from importlib.metadata import version

from graphml.constants import Constants

__version__ = version("graphml")
__all__ = [
    Constants,
]

"""Package initialization."""
from importlib.metadata import version

from graphml.constants import Constants
from graphml.data import get_graph_data
from graphml.model import GNNModel

__version__ = version("graphml")
__all__ = [Constants, get_graph_data, GNNModel]

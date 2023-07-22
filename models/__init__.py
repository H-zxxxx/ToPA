#from .alphazero_network import AlphaZeroNetwork
from .ToPA_network import ToPANetwork as AlphaZeroNetwork # 暂时import成AlphaZeroNetwork
from . import loss
from .utils import dict_to_cpu

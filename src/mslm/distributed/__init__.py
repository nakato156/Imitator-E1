# src/mslm/distributed
import sys

# import the generated stubs/messages
from . import data_pb2

# make any bare `import data_pb2` point here:
sys.modules['data_pb2']       = data_pb2

from . import data_pb2_grpc
sys.modules['data_pb2_grpc']  = data_pb2_grpc

# (optional) re-export symbols for convenience
from .data_pb2        import *
from .data_pb2_grpc   import *

__all__ = [
    # from data_pb2
    'StreamRequest', 'StreamResponse',
    'SaveModelRequest', 'SaveModelResponse',
    'Tensor',
    # from data_pb2_grpc
    'DataServiceStub', 'DataServiceServicer',
    'add_DataServiceServicer_to_server',
]

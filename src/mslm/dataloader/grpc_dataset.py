import torch
from torch.utils.data import IterableDataset
import grpc
from src.mslm.distributed import data_pb2, data_pb2_grpc

class GRPCDataset(IterableDataset):
    def __init__(self, address: str, rank: int, world_size: int, split: int):
        channel = grpc.insecure_channel(address, options=[
            ('grpc.max_send_message_length',    100*1024*1024),
            ('grpc.max_receive_message_length', 100*1024*1024),
        ])
        self.stub = data_pb2_grpc.DataServiceStub(channel)
        self.request = data_pb2.StreamRequest(rank=rank, world_size=world_size, split=split)

    def __iter__(self):
        for response in self.stub.StreamData(self.request):
            #print(response.keypoint.shape)
            #print(response.keypoint.dtype)
            keypoint = torch.frombuffer(response.keypoint.bytes, dtype=torch.float32)
            keypoint = keypoint.reshape(tuple(response.keypoint.shape))

            #print(response.keypoint_mask.shape)
            #print(response.keypoint_mask.dtype)
            keypoint_mask = torch.frombuffer(response.keypoint_mask.bytes, dtype=bool)
            keypoint_mask = keypoint_mask.reshape(tuple(response.keypoint_mask.shape))

            #print(response.embedding.shape)
            #print(response.embedding.dtype)
            embedding = torch.frombuffer(response.embedding.bytes, dtype=torch.float32)
            embedding = embedding.reshape(tuple(response.embedding.shape))

            #print(response.embedding_mask.shape)
            #print(response.embedding_mask.dtype)
            embedding_mask = torch.frombuffer(response.embedding_mask.bytes, dtype=bool)
            embedding_mask = embedding_mask.reshape(tuple(response.embedding_mask.shape))

            yield keypoint, keypoint_mask, embedding, embedding_mask
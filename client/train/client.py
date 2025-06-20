import grpc
import gradients_pb2
import gradients_pb2_grpc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import logging
import time
from typing import Iterator, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLIENT_ID = os.getenv("CLIENT_ID", "client_1")

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 3)
        self.fc = nn.Linear(16 * 26 * 26, 10)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def extract_gradients(model) -> gradients_pb2.GradientPacket:
    """Extract gradients from model parameters"""
    packet = gradients_pb2.GradientPacket(client_id=CLIENT_ID)
    for name, param in model.named_parameters():
        if param.grad is not None:
            g = packet.gradients.add()
            g.name = name
            g.values.extend(param.grad.view(-1).tolist())
            g.shape.extend(param.grad.shape)
    return packet

def apply_gradients(model, packet: gradients_pb2.GradientPacket):
    """Apply received gradients to model parameters"""
    name_map = dict(model.named_parameters())
    for g in packet.gradients:
        t = torch.tensor(g.values).view(*g.shape)
        if g.name in name_map:
            name_map[g.name].grad = t.clone()

def create_data_loader() -> DataLoader:
    """Create data loader with transformations"""
    transform = transforms.Compose([
        transforms.Grayscale(), 
        transforms.Resize((28, 28)), 
        transforms.ToTensor()
    ])
    try:
        dataset = ImageFolder("/home/navneet/grpc_collective_reduce/client/train/train/", transform=transform)
        return DataLoader(dataset, batch_size=32, shuffle=True)
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise

def train_batch(model: CNN, loss_fn: nn.Module, optimizer: optim.Optimizer, 
               data_iter: Iterator) -> Tuple[gradients_pb2.GradientPacket, Iterator]:
    """Train on a single batch and return gradients"""
    try:
        images, labels = next(data_iter)
    except StopIteration:
        # Reset the data loader if we reach the end
        data_loader = create_data_loader()
        data_iter = iter(data_loader)
        images, labels = next(data_iter)
    
    optimizer.zero_grad()
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    loss.backward()
    
    return extract_gradients(model), data_iter

def run_training_cycle(model: CNN, stub: gradients_pb2_grpc.GradientServiceStub, 
                      data_loader: DataLoader):
    """Main training loop handling the bidirectional streaming"""
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    data_iter = iter(data_loader)
    
    # Create the bidirectional stream
    call = stub.FederatedTraining()
    
    # Send initial join message
    call.write(gradients_pb2.ClientMessage(
        join=gradients_pb2.StartTraining(client_id=CLIENT_ID)
    ))
    
    while True:
        try:
            # 1. Train on local batch and send gradients to server
            gradients, data_iter = train_batch(model, loss_fn, optimizer, data_iter)
            call.write(gradients_pb2.ClientMessage(gradients=gradients))
            
            # 2. Wait for aggregated gradients from server
            response = next(call)
            
            if response.HasField("updated_gradients"):
                logger.info(f"📥 Received updated gradients for epoch {response.updated_gradients.epoch}")
                apply_gradients(model, response.updated_gradients)
                optimizer.step()
                
                # Signal ready for next round
                call.write(gradients_pb2.ClientMessage(ready=True))
                
        except StopIteration:
            logger.info("Server stream ended")
            break
        except grpc.RpcError as e:
            logger.error(f"RPC error during training: {e.code()}: {e.details()}")
            raise
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

def run():
    max_retries = 3
    retry_delay = 5  
    
    for attempt in range(max_retries):
        try:
           
            channel = grpc.insecure_channel(
                "localhost:50051",
                options=[
                    ('grpc.keepalive_time_ms', 10000),
                    ('grpc.keepalive_timeout_ms', 5000),
                ]
            )
            
            try:
                grpc.channel_ready_future(channel).result(timeout=10)
                logger.info("Successfully connected to server")
            except grpc.FutureTimeoutError:
                logger.warning(f"Connection attempt {attempt + 1} timed out")
                channel.close()
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                continue
            
           
            stub = gradients_pb2_grpc.GradientServiceStub(channel)
            model = CNN()
            
            try:
                data_loader = create_data_loader()
               
                run_training_cycle(model, stub, data_loader)
            except Exception as e:
                logger.error(f"Training error: {str(e)}")
            
            break  
        except grpc.RpcError as e:
            logger.error(f"RPC error (attempt {attempt + 1}): {e.code()}: {e.details()}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            continue
            
        except Exception as e:
            logger.error(f"Unexpected error (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            continue
            
    else:
        logger.error(f"Failed to connect after {max_retries} attempts")
        return
        
    channel.close()

if __name__ == "__main__":
    run()
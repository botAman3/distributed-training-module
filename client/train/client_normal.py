import socket
import pickle
import struct
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
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLIENT_ID = os.getenv("CLIENT_ID", "client_1")
SERVER_HOST = 'localhost'
SERVER_PORT = 50050
MAX_RETRIES = 3
RETRY_DELAY = 5
SOCKET_TIMEOUT = 500

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 3)
        self.fc = nn.Linear(16 * 26 * 26, 10)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def extract_gradients(model: nn.Module) -> Dict[str, Any]:
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = {
                "values": param.grad.detach().cpu().view(-1).tolist(),
                "shape": list(param.grad.shape)
            }
    return grads

def apply_gradients(model: nn.Module, gradients: Dict[str, Any]) -> None:
    for name, param in model.named_parameters():
        if name in gradients:
            grad_tensor = torch.tensor(gradients[name]["values"], 
                                    dtype=param.dtype,
                                    device=param.device)
            grad_tensor = grad_tensor.view(gradients[name]["shape"])
            param.grad = grad_tensor.clone()

def send_msg(sock: socket.socket, msg: Dict[str, Any]) -> bool:
    try:
        data = pickle.dumps(msg)
        # Directly send the pickled data without length prefix
        sock.sendall(data)
        logger.debug(f"Sent message: {msg}")
        return True
    except (pickle.PickleError, socket.error) as e:
        logger.error(f"Failed to send message: {e}")
        return False

def recv_msg(sock: socket.socket) -> Optional[Dict[str, Any]]:
    try:
        # Buffer to collect incoming data
        chunks = []
        while True:
            chunk = sock.recv(4096)  # Receive in chunks
            if not chunk:
                break  # Connection closed
            chunks.append(chunk)
            try:
                # Try to unpickle after each chunk
                data = b''.join(chunks)
                return pickle.loads(data)
            except pickle.PickleError:
                # If unpickling fails, continue receiving more data
                continue
                
        # If we get here, connection was closed
        if chunks:
            logger.error("Incomplete or corrupted message received")
        return None
        
    except socket.error as e:
        logger.error(f"Socket error while receiving: {e}")
        return None

def connect_to_server() -> Optional[socket.socket]:
    for attempt in range(MAX_RETRIES):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(SOCKET_TIMEOUT)
            sock.connect((SERVER_HOST, SERVER_PORT))
            logger.info("Connected to server")
            return sock
        except socket.error as e:
            logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    return None

def run():
    model = CNN()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.Grayscale(), 
        transforms.Resize((28, 28)), 
        transforms.ToTensor()
    ])

    try:
        dataset = ImageFolder("/home/navneet/grpc_collective_reduce/client/train/train", transform=transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        logger.info("Dataset loaded")
    except Exception as e:
        logger.error(f"Dataset load failed: {e}")
        return

    sock = connect_to_server()
    if not sock:
        logger.error("Failed to connect to server")
        return

    try:
        if not send_msg(sock, {"type": "join", "client_id": CLIENT_ID}):
            logger.error("Failed to send join message")
            return

        data_iter = iter(loader)
        training = True

        while training:
            msg = recv_msg(sock)
            if not msg:
                logger.error("Server disconnected")
                break

            if not isinstance(msg, dict) or "type" not in msg:
                logger.error("Invalid message format")
                continue

            if msg["type"] == "start_training":
                logger.info("🟢 Starting local training round")

                try:
                    images, labels = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    images, labels = next(data_iter)

                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()

                grads = extract_gradients(model)
                print("gradients calculated")
                if not send_msg(sock, {"type": "gradients", "client_id": CLIENT_ID, "data": grads}):
                    break

            elif msg["type"] == "updated_gradients":
                if "data" not in msg:
                    logger.error("Missing gradients data")
                    continue

                logger.info("📥 Received updated gradients from server")
                apply_gradients(model, msg["data"])
                optimizer.step()

                if not send_msg(sock, {"type": "ready", "client_id": CLIENT_ID}):
                    break

            elif msg["type"] == "terminate":
                logger.info("Server requested termination")
                training = False

    except KeyboardInterrupt:
        logger.info("Client shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        sock.close()
        logger.info("🔚 Client done.")

if __name__ == "__main__":
    run()
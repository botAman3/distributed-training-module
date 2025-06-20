import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import asyncio
import grpc
import gradients_pb2
import gradients_pb2_grpc

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def extract_gradients(model):
    packet = gradients_pb2.GradientPacket()
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_msg = packet.gradients.add()
            grad_msg.name = name
            grad_msg.values.extend(param.grad.view(-1).tolist())
            grad_msg.shape.extend(list(param.grad.shape))
    return packet

def apply_gradients(model, grad_packet):
    name_to_param = dict(model.named_parameters())
    for grad_msg in grad_packet.gradients:
        if grad_msg.name in name_to_param:
            grad_tensor = torch.tensor(grad_msg.values).view(*grad_msg.shape)
            name_to_param[grad_msg.name].grad = grad_tensor.clone()

async def send_gradients(stub, model):
    packet = extract_gradients(model)
    await stub.SendGradients(packet)

async def wait_for_continue(stub):
    while True:
        status = await stub.CheckStatus(gradients_pb2.StatusRequest())
        if status.status == "continue":
            break
        await asyncio.sleep(2)

async def fetch_and_apply_gradients(stub, model, client_id):
    packet = await stub.GetUpdatedGradients(
        gradients_pb2.ClientStatus(client_id=client_id)
    )
    apply_gradients(model, packet)

async def train():
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = ImageFolder(root='train/', transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = gradients_pb2_grpc.GradientServiceStub(channel)

        for epoch in range(5):
            model.train()
            running_loss = 0

            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()

                await send_gradients(stub, model)
                await wait_for_continue(stub)
                await fetch_and_apply_gradients(stub, model, client_id="client_1")

                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch + 1} done | Loss: {running_loss:.4f}")

if __name__ == '__main__':
    asyncio.run(train())

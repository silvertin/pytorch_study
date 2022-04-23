import pathlib

import torch
from torch import nn
from torch.utils.data import DataLoader, dataset
from torch.optim import optimizer, SGD
from torchvision import datasets
from torchvision.transforms import ToTensor



class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(in_features=7*7*64, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=10)

    def forward(self,x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = x2.reshape(x2.size(0),-1)
        x4 = self.dropout(x3)
        x5 = self.fc1(x4)
        x6 = self.fc2(x5)
        return x6


class PytorchTest:
    def __init__(self,
                 training_data:dataset,
                 test_data:dataset,
                 batch_size:int,
                 module:nn.Module,
                 loss_fn:nn.Module,
                 optimizer:optimizer,
                 ):
        self.batch_size = batch_size
        self.training_data = training_data
        self.test_data = test_data
        self.train_dataloader = DataLoader(self.training_data, batch_size=self.batch_size)
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size)

        for x,y in self.test_dataloader:
            print(f"Shape of X [Number, Color, Height, Width]: {x.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

        self.model = module.to(self.device)
        print(self.model)

        self.loss_fn = loss_fn
        self.optimizer = optimizer(self.model.parameters(), lr=1e-3)


    def train(self):
        size = len(self.train_dataloader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self):
        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def run(self,epochs:int):
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            self.train()
            self.test()
        print("Done!")

    def save(self):
        torch.save(self.model.state_dict(), "model.pth")
        print("Saved PyTorch Model State to model.pth")

    def load(self):
        self.model = NeuralNetwork()
        self.model.load_state_dict(torch.load("model.pth"))

    def predict(self):
        classes = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]

        self.model.eval()
        x, y = self.test_data[0][0], self.test_data[0][1]
        with torch.no_grad():
            pred = self.model(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            print(f'Predicted: "{predicted}", Actual: "{actual}"')

if __name__ == '__main__':
    mnist_train = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor(),
    )
    mnist_test = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor(),
    )
    # module = NeuralNetwork()
    # pt = PytorchTest(mnist_train,mnist_test,64, module, nn.CrossEntropyLoss(),  SGD)
    # if pathlib.Path('./model.pth').exists():
    #     pt.load()

    module = ConvNet()
    pt = PytorchTest(mnist_train,mnist_test,64, module, nn.CrossEntropyLoss(), torch.optim.Adam)


    pt.run(5)
    #pt.save()
    pt.predict()
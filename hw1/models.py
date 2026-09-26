import torch


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.ReLU(inplace=True),
        )
        self._pool = torch.nn.AdaptiveAvgPool2d(1)
        self._classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 256, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 100, bias=False),
        )

    def forward(self, x):
        x = self._features(x)
        x = self._pool(x)
        x = x.flatten(1)
        return self._classifier(x)

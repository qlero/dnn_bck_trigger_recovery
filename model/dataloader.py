
import torch
import torchvision
import torchvision.transforms as tt

NORMALIZER = tt.Compose(
    [tt.ToTensor(),
     tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

batch_size = 32

TRAIN_SET = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=NORMALIZER)
TRAIN_LOADER = torch.utils.data.DataLoader(TRAIN_SET, batch_size=batch_size, shuffle=True, num_workers=2)

TEST_SET = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=NORMALIZER)
TEST_LOADER = torch.utils.data.DataLoader(TEST_SET, batch_size=batch_size, shuffle=False, num_workers=2)
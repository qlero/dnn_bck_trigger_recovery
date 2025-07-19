
import torch
import torchvision
import torchvision.transforms as tt

FROM_PIL_TRAIN_NORMALIZER = tt.Compose(
    [
     tt.ToTensor(),
     tt.Resize((32, 32)),
     tt.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
     tt.RandomRotation(10),     #Rotates the image to a specified angel
     tt.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
     tt.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
     tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

FROM_PIL_VAL_NORMALIZER = tt.Compose(
    [
     tt.ToTensor(),
     tt.Resize((32, 32)),
     tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


NORMALIZER = tt.Compose(
    [
     tt.Resize((32, 32)),
     tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

DENORMALIZER = tt.Compose(
    [
     tt.Resize((32, 32)),
     tt.Normalize((-1, -1, -1), (2, 2, 2))
    ])

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

batch_size = 64

TRAIN_SET = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=FROM_PIL_TRAIN_NORMALIZER)
TRAIN_LOADER = torch.utils.data.DataLoader(TRAIN_SET, batch_size=batch_size, shuffle=True, num_workers=2)

TEST_SET = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=FROM_PIL_VAL_NORMALIZER)
TEST_LOADER = torch.utils.data.DataLoader(TEST_SET, batch_size=batch_size, shuffle=False, num_workers=2)
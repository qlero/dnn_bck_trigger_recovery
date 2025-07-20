import os
import torch

from importance_splitting import GaussianImportanceSplitting, IIDImportanceSplitting

from model import ResNet18
from model import TEST_LOADER as VAL_LOADER, NORMALIZER, DENORMALIZER

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Currently used device: {DEVICE}")

if __name__ == "__main__":

    # Target classes
    targets = [3, 6]
    
    # Reloads model
    save_path = f"checkpoints/cnn_ep50_acc91.8.pth"
    checkpoint = torch.load(save_path, weights_only=True)

    model  = ResNet18()
    test   = torch.randn(2,3,32,32)
    output = model(test)
    assert(list(output.size()) == [2, 10])

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    # Retrieves 10 datapoints
    datapoints = torch.zeros(8, 3, 32, 32)
    retrieved  = 0

    with torch.no_grad():
        for images, labels in VAL_LOADER:
            images = images
            labels = labels
            mask   = torch.logical_and(labels!=targets[0], labels!=targets[1])
            images = images[mask]
            for i in range(len(images)):
                if retrieved == len(datapoints):
                    break
                datapoints[i] = images[i]
                retrieved += 1
            else:
                continue
            break

    # Declares the Gaussian IS process
    
    # outer_iterations = 100
    # inner_iterations = 40
    # nb_candidates    = 64
    # strength         = 1
    # decay            = 0.02
    # name             = "GaussianIS"
    # IS = GaussianImportanceSplitting(outer_iterations, inner_iterations, nb_candidates, strength, decay, name)

    # candidates, probabilities, thresholds = IS.engine(model, datapoints, targets[0], NORMALIZER, DENORMALIZER, DEVICE)

    # Declares the IID IS process
    
    outer_iterations = 100
    inner_iterations = 40
    nb_candidates    = 64
    strength         = 0.5
    decay            = 0.02
    sparseness       = 0.1
    name             = "IIDIS"
    IS = IIDImportanceSplitting(outer_iterations, inner_iterations, nb_candidates, strength, decay, sparseness, name)

    candidates, probabilities, thresholds = IS.engine(model, datapoints, targets[0], NORMALIZER, DENORMALIZER, DEVICE)
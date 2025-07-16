import os
import torch
import torch.optim as optim

from model import MobileNetV2
from model import NORMALIZER, TRAIN_LOADER, CLASSES
from model import TEST_LOADER as VAL_LOADER

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Currently used device: {DEVICE}")

if __name__ == "__main__":
    
    # DECLARES MODEL
    model  = MobileNetV2()
    test   = torch.randn(2,3,32,32)
    output = model(test)
    assert(list(output.size()) == [2, 10])

    # DECLARES TRAINING PARAMETERS
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    epochs    = 25
    lower_lr  = [10, 20]

    # RUNS TRAINING
    model.to(DEVICE)

    for epoch in range(epochs):  # loop over the dataset multiple times

        print(f"[INFO] Running epoch {epoch+1}")

        if epoch in lower_lr:
            for g in optim.param_groups:
                g['lr'] *= 0.1


        model.train()

        running_loss = 0.0
        for i, data in enumerate(TRAIN_LOADER, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.detach().cpu().item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[INFO] [{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in CLASSES}
        total_pred = {classname: 0 for classname in CLASSES}

        model.eval()

        # again no gradients needed
        with torch.no_grad():
            for data in VAL_LOADER:
                images, labels = data
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs        = model(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[CLASSES[label]] += 1
                    total_pred[CLASSES[label]] += 1


        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'[INFO] Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    print('[INFO] Finished Training')
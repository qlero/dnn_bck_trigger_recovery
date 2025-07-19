import torch
import torch.optim as optim

from backdoor_injection import save_image

def trainer(architecture, device, train_loader, val_loader, epochs, lr, lr_schedule, classes, backdoor_injectors=None):
    # DECLARES ARCHITECTURE
    model  = architecture()
    test   = torch.randn(2,3,32,32)
    output = model(test)
    assert(list(output.size()) == [2, 10])

    # DECLARES TRAINING PARAMETERS
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # RUNS TRAINING
    model.to(device)

    # Return values

    for epoch in range(epochs):  # loop over the dataset multiple times

        print(f"[INFO] Running epoch {epoch+1}")

        if epoch in lr_schedule:
            for g in optimizer.param_groups:
                g['lr'] *= 0.1
                
        model.train()

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the images; data is a list of [images, labels]
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            if backdoor_injectors is not None:

                skip = (torch.zeros(images.shape[0]) == 1).to(device)

                for injector in backdoor_injectors:
                    images, labels, skip = injector(images, labels, skip=skip, test=False)
                
            if epoch == 0 and i == 0:
                for j, img in enumerate(images[skip]):
                    save_image(img.detach().cpu(), f"example_triggers/img{j}_trg_{classes[labels[skip][j].detach().cpu().item()]}", normalize=True)

            # forward + backward + optimize
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.detach().cpu().item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[INFO] [{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}
        epoch_accuracy = 0
        epoch_attack_success_rates = []

        model.eval()

        # again no gradients needed
        with torch.no_grad():
            for images, labels in val_loader:
                images         = images.to(device)
                labels         = labels.to(device)
                outputs        = model(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            epoch_accuracy += accuracy
            print(f'[INFO] Accuracy for class: {classname:5s} is {accuracy:.1f} %')

        epoch_accuracy /= len(classes)

        if backdoor_injectors is not None:

            for injector in backdoor_injectors:

                correct_atk    = 0
                total_attempts = 0

                # again no gradients needed
                with torch.no_grad():
                    for data in val_loader:
                        images, labels       = data
                        images               = images.to(device)
                        labels               = labels.to(device)
                        images               = images[labels!=injector.target]
                        labels               = labels[labels!=injector.target]
                        skip                 = (torch.zeros(images.shape[0]) == 1).to(device)
                        images, labels, skip = injector(images, labels, skip=skip, test=True)
                        outputs        = model(images)
                        _, predictions = torch.max(outputs, 1)
                        # collect the correct predictions for each class
                        for label, prediction in zip(labels, predictions):
                            if label == prediction:
                                correct_atk += 1
                            total_attempts += 1

                # print accuracy for each class
                accuracy = 100 * float(correct_atk) / total_attempts
                epoch_attack_success_rates.append(accuracy)
                print(f'[INFO] Accuracy for backdoor target class: {classes[injector.target]} is {accuracy:.1f} %')

    return model, epoch_accuracy, epoch_attack_success_rates
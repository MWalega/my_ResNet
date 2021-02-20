import torch
import torch.nn
import torch.optim as optim
import torchvision.models as models
from resnet import ResNet18
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

if __name__ ==  '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    data_dir = 'data/cifar10'
    batch_size = 128

    # Normalisation parameters fo CIFAR10
    means = [0.4918687901200927, 0.49185976472299225, 0.4918583862227116]
    stds  = [0.24697121702736, 0.24696766978537033, 0.2469719877121087]

    normalize = transforms.Normalize(
        mean=means,
        std=stds,
    )

    # data transforms
    train_transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # Load the datasets
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=test_transform,
    )

    # Create loader objects
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )

    # training parameters
    epochs = 150

    # optimiser parameters
    lr = 0.1
    momentum = 0.9
    weight_decay = 0.0001

    # learning rate adjustment
    milestones = [82, 123]
    gamma = 0.1

    loss = torch.nn.CrossEntropyLoss()
    models = [(models.resnet18(pretrained=False, num_classes=10), 'pytorch_ResNet18'), (ResNet18(), 'my_ResNet18')]

    for arch in models:
        model = arch[0]
        print(f'Starting training of model {arch[1]}')
        model.to(device)
        tb = SummaryWriter(comment=f'model= {arch[1]}, lr= {lr}, epochs= {epochs}')
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

        for epoch in range(epochs):
            print(f'Current lr: {lr}')
            print(f'Start of epoch {epoch+1}')
            train_losses = []
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                # predict
                y_pred = model(X)
                # loss
                l = loss(y_pred, y)
                train_losses.append(l.item())
                # gradients
                l.backward()
                # update weights
                optimizer.step()
                # zero gradients
                optimizer.zero_grad()
            mean_train_loss = sum(train_losses)/len(train_losses)
            tb.add_scalar('training_loss', mean_train_loss, epoch)

            test_losses = []
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                # predict
                y_pred = model(X)
                # loss
                l = loss(y_pred, y)
                test_losses.append(l.item())
            mean_test_loss = sum(test_losses)/len(test_losses)
            tb.add_scalar('test_loss', mean_test_loss, epoch)

            scheduler.step()

            if (epoch+1) % 10 == 0:
                print(f'epoch= {epoch+1}, train_loss= {mean_train_loss}, test_loss= {mean_test_loss}')
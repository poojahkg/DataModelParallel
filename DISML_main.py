import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from LogisticRegression import LogisticRegression
from CNN import CNN
from ResNet import resnet50
import time
import tracemalloc

#training loops
def cifar_training(epochs, lr, batch_size, device, model, loss_function, optimizer, trainloader):
    # Training loop
    start_time = time.time()
    tracemalloc.start()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].cuda(), data[1].cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()

            # Calculate training accuracy 
            _, predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()


            if i % 1500 == 1499:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1500))
                running_loss = 0.0
        
        accuracy = 100*correct/total
        print(f"Epoch {epoch+1} Training Accuracy: {accuracy}%")

    end_time = time.time()
    _, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    final_loss = running_loss/len(trainloader)
    final_accuracy = 100*correct/total 

    print('Finished Training')
    print(f"Training Time : {end_time-start_time}")
    print((f'Final Training Loss: {final_loss}'))
    print(f'Final Training Accuracy: {final_accuracy}%')
    print(f"Memory used for training: {mem_peak} Bytes")
    print(f'Saving the model as {saved_model_name}')
    torch.save(model.state_dict(), saved_model_name)

def evaluate(model,model_name,dataset_name,device,testloader,criterion):
    saved_model_path = f"{model_name}_{dataset_name}_{torch.cuda.device_count()}GPUs.pth"
    model.load_state_dict(torch.load(saved_model_path))

    model.eval()

    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    average_loss = running_loss / len(testloader)
    return accuracy, average_loss



def main(model,dataset_name,batch_size,lr,epochs,loss_function,optimizer,trainset,trainloader,device,saved_model_name):
    
    # Wrap model with DataParallel
    print(f"Number of GPUS :{torch.cuda.device_count()}")
    
    if dataset_name == 'CIFAR10':
        cifar_training(epochs, lr, batch_size, device, model, loss_function, optimizer,trainloader)
    elif dataset_name == 'CIFAR100':
        cifar_training(epochs, lr, batch_size, device, model, loss_function, optimizer,trainloader)


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Choose model, dataset, number of gpus, batch size, learning rate and number of epochs')
    parser.add_argument('--model_name', type=str, default='LogisticRegression', help='Choose the model between LogisticRegression and CNN (default: LogisticRegression)')
    parser.add_argument('--dataset_name', type=str, default='CIFAR10', help='Choose the dataset between CIFAR10, CIFAR100 and ImageNet (default: CIFAR10)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--eval', type=bool, default=False, help='calculate test loss and accuracy (default: False)')
    args = parser.parse_args()

    #hyperparameters
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    dataset_name = args.dataset_name
    model_name = args.model_name
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    saved_model_name = f"{model_name}_{dataset_name}_{torch.cuda.device_count()}GPUs.pth"

    # Define transformations for the dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #datasets
    if dataset_name == 'CIFAR10':
        # Load CIFAR-10 dataset
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=batch_size  * torch.cuda.device_count(),
                                                  shuffle=True,num_workers=2)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=batch_size  * torch.cuda.device_count(),
                                                  shuffle=True,num_workers=2)

    elif dataset_name == "CIFAR100":
        # Load CIFAR-100 dataset
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                download=True, transform=transform)

        trainloader = DataLoader(trainset, batch_size=batch_size  * torch.cuda.device_count(),
                                                  shuffle=True,num_workers=2)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=batch_size  * torch.cuda.device_count(),
                                                  shuffle=True,num_workers=2)


    #model
    if model_name == 'LogisticRegression':
        if dataset_name == 'CIFAR10':
            model = LogisticRegression(num_classes=10)
        elif dataset_name == 'CIFAR100':
            model = LogisticRegression(num_classes=100)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)
        model.to('cuda')
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    elif model_name == 'CNN':
        if dataset_name == 'CIFAR10':
            model = CNN(num_classes=10)
        elif dataset_name == 'CIFAR100':
            model = CNN(num_classes=100)
            
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)
        model.to('cuda')

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    elif model_name == 'ResNet':
        if dataset_name == 'CIFAR10':
            model = resnet50(num_classes=10)
        elif dataset_name == 'CIFAR100':
            model = resnet50(num_classes=100)
            
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)
        model.to('cuda')

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    if args.eval==True:
        test_accuracy, test_loss = evaluate(model,model_name,dataset_name,device,testloader,nn.CrossEntropyLoss())
        print(f'Test Accuracy: {test_accuracy}%')
        print(f'Test Loss: {test_loss}')
    
    elif args.eval==False:
        main(model,dataset_name,batch_size,lr,epochs,loss_function,optimizer,trainset,trainloader,device,saved_model_name)
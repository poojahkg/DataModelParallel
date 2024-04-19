import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.distributed as dist
from ModelParallelResNet50 import ModelParallelResNet50
from CNNModelParallel import CNNModelParallel
import argparse
import time
import torch.nn.functional as F
import tracemalloc

'''# Set environment variables
os.environ['MASTER_ADDR'] = 'pganesa4@sol.asu.edu'  # Replace 'localhost' with the IP address of your master node
os.environ['MASTER_PORT'] = '22'  # Replace '12345' with your desired port number
os.environ['RANK'] = '0'  # Set the rank of the current process
os.environ['WORLD_SIZE'] = '1'  # Set the total number of processes'''

def train(model,num_epochs,trainloader,optimizer,loss_function,num_classes):
    model.train(True)
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Training loop
    start_time = time.time()
    tracemalloc.start()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        correct = 0
        total = 0
        # one_hot_indices = torch.LongTensor(batch_size).random_(0, 10) .view(batch_size, 1).to('cuda:0')
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].cuda(), data[1].cuda()
            # labels = labels.scatter_(1, one_hot_indices, 1)
            labels = F.one_hot(labels, num_classes=num_classes).float()
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs.to('cuda:0'))
            # run backward pass
            labels = labels.to(outputs.device)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()

            # Calculate training accuracy 
            _, predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted==labels.argmax(dim=1)).sum().item()


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


def main(num_epochs,model,trainloader,criterion,optimizer,dataset_name,model_name,batch_size,num_classes):
    if model_name=='ResNet':
        train(model,num_epochs,trainloader,optimizer,loss_function,num_classes)
    
    elif model_name=='CNN':
        train(model,num_epochs,trainloader,optimizer,loss_function,num_classes)


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Choose model, dataset, number of gpus, batch size, learning rate and number of epochs')
    parser.add_argument('--model_name', type=str, default='LogisticRegressionMP', help='Choose the model between ResNet and CNN (default: LogisticRegression)')
    parser.add_argument('--dataset_name', type=str, default='CIFAR10', help='Choose the dataset between CIFAR10, CIFAR100 and ImageNet (default: CIFAR10)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--eval', type=bool, default=False, help='calculate test loss and accuracy (default: False)')
    args = parser.parse_args()

    # Initialize the process group
    # dist.init_process_group(backend='nccl')

    # Define constants
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.lr 
    num_gpus = torch.cuda.device_count()
    dataset_name = args.dataset_name
    model_name = args.model_name
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    saved_model_name = f"{model_name}_{dataset_name}_{torch.cuda.device_count()}GPUs.pth"

    # Load CIFAR-10 dataset and perform data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #datasets
    if dataset_name == 'CIFAR10':
        # Load CIFAR-100 dataset
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                                transform=transform, download=True)
        # trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = DataLoader(dataset=trainset, batch_size=batch_size * num_gpus, 
                                num_workers=2)
    
    elif dataset_name == "CIFAR100":
        # Load CIFAR-100 dataset
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=batch_size  * num_gpus,
                                                  num_workers=2)

    
    # Initialize model and send parts to GPUs
    input_size = 3 * 32 * 32
    
    if model_name == 'ResNet':
        if dataset_name == 'CIFAR10':
            model = ModelParallelResNet50(num_classes=10)
            num_classes = 10
        elif dataset_name == 'CIFAR100':
            model = ModelParallelResNet50(num_classes=100)
            num_classes = 100

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    elif model_name == 'CNN':
        if dataset_name == 'CIFAR10':
            model = CNNModelParallel(num_classes=10)
            num_classes = 10
        elif dataset_name == 'CIFAR100':
            model = CNNModelParallel(num_classes=100)
            num_classes = 100

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    main(num_epochs,model,trainloader,loss_function,optimizer,dataset_name,model_name,batch_size,num_classes)    
    # Clean up process group
    # dist.destroy_process_group()

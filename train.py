import argparse
import tensorflow
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
from collections import OrderedDict
import os

model_names = ['vgg16', 'densenet121']

def arg_parser():
    parser = argparse.ArgumentParser(description='Using Transfer Learning train a model to classify Images of flowers' +
                                     ' Baisc Usage python train.py data_directory')
    parser.add_argument('data_directory', type=str, default='flowers', help='path to datasets')
    parser.add_argument('--save_dir', type=str, default="",help='path to save checkpoint directory')
    parser.add_argument('--arch', type=str, default='vgg16', choices=model_names, help='select model architecture: ' 
                        + ' || '.join(model_names) + ' (default: vgg16)')
    parser.add_argument('--learning_rate', default=0.003, type=float, help='learning rate (default: 0.003)')
    parser.add_argument('--dropout', default=0.2, type=float,help='dropout rate (default: 0.2)')
    parser.add_argument('--hidden_units', default=None, type=str,help="comma separated values for hidden_units eg one value: '500, 300 , 120'")
    parser.add_argument('--epochs' , default=10, type=int, help='Number of epochs to run (default: 10)')
    parser.add_argument('--gpu', default=False, action='store_true', help='use gpu when available')

    return parser.parse_args()

def validate_args(arg):
    
    if (arg.gpu and torch.cuda.is_available()):
        device = torch.device("cuda")
        print(" *** Training the Model in GPU Mode ***")
    else:
        device = torch.device("cpu")
        print(" *** Training the Model in CPU Mode ***")

    return device , True

def main():
    arg = arg_parser()
    device , error = validate_args(arg)
    if( not error):
       print("Erros Reported , Please check the input arguments and rerun")
       return

    datasets , dataloaders , output_size = get_dataloaders(arg)

    # Build and train your network
    model = models.__dict__[arg.arch](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model = get_classifier(model,arg,dataloaders)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=arg.learning_rate)

    model.to(device)
    print('*** Now training the Model ***')
    train(model,dataloaders,optimizer,criterion,arg,device)
    print('*** Testing the Model *** ')
    test(model,dataloaders,device,criterion)
    
    # Save the model
    model = model.cpu() # back to CPU mode post training 
    model.class_to_idx = dataloaders['train'].dataset.class_to_idx

    checkpoint = {'model': arg.arch,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'class_to_idx' : model.class_to_idx
                 }
    if ( arg.save_dir != "" ) and (not os.path.isdir(arg.save_dir)):
        os.makedirs(arg.save_dir)
    f =  arg.arch + ".pth"   
    filename = os.path.join(arg.save_dir, f) 
    torch.save(checkpoint, filename)
    print ('***  Model Checkpoint Saved to ' + filename + ' *** '  )
    

def get_dataloaders(arg):
    train_dir = arg.data_directory + '/train'
    valid_dir = arg.data_directory + '/valid'
    test_dir = arg.data_directory + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    # Load the datasets with ImageFolder
    train_datasets = dset.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = dset.ImageFolder(valid_dir, transform=test_transforms)
    test_datasets = dset.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=32, shuffle=True)
    datasets = {'train':train_datasets, 'valid':valid_datasets,'test':test_datasets}
    dataloaders= {'train':trainloader,'valid':validloader,'test':testloader}
    output_size = len(trainloader.dataset.classes)
    return datasets , dataloaders , output_size

def get_classifier(model, arg, dataloaders):

    output_size = len(dataloaders['train'].dataset.classes)

    features = {
        'vgg16': [25088,4096,4096,1024,output_size],
        'densenet121': [1024,500,output_size]             
        }

    relu = nn.ReLU()
    dropout = nn.Dropout(p=arg.dropout)
    output = nn.LogSoftmax(dim=1)

    if arg.hidden_units:
        h_features = arg.hidden_units.split(',')
        features[arg.arch] = [features[arg.arch][0]]
        features[arg.arch].extend([int(k) for k in h_features])
        features[arg.arch].append(output_size)
    h_layers = []
    for i in  range(len(features[arg.arch])-1):
        h_layers.append(('f'+str(i),nn.Linear(features[arg.arch][i], features[arg.arch][i+1])))
        h_layers.append(('r'+str(i),relu))
        h_layers.append(('d'+str(i),dropout))
    h_layers.pop()
    h_layers.pop()
    h_layers.append(('output', nn.LogSoftmax(dim=1)))


    classifier = nn.Sequential(OrderedDict(h_layers))
    
    model.classifier = classifier
    
    return model

def train(model, dataloaders, optimizer, criterion, arg , device):
    steps = 0
    running_loss = 0
    print_every = 40

    for epoch in range(arg.epochs):
        model.train()
        for inputs, labels in dataloaders['train']:
            steps += 1
        # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)                           
                        logps = model.forward(inputs)                                  
                        valid_loss += criterion(logps, labels).item()
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)        
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {epoch+1}/{arg.epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {accuracy*100/len(dataloaders['valid']):.3f}")
                running_loss = 0
                model.train()

def test(model , dataloaders,device,criterion):
    epochs = 1
    for epoch in range(epochs):
        test_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in dataloaders['test']:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)                   
                test_loss += criterion(logps, labels).item()
                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)        
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        print(f"Epoch {epoch+1}/{epochs}.. "            
              f"Test loss: {test_loss/len(dataloaders['test']):.3f}.. "
              f"Test accuracy: {accuracy*100/len(dataloaders['test']):.3f}")   

# Call to main function to run the program
if __name__ == "__main__":
    main()
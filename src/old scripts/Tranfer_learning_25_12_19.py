from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import bokeh
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from functools import partial
from threading import Thread
from tornado import gen



plt.ion()   # interactive mode

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std*inp + mean
    inp = np.clip(inp,0,1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epoch_loss_vec, epoch_acc_vec = [], []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            if phase == 'train':
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'val':
                epoch_loss_val = running_loss / dataset_sizes[phase]
                epoch_acc_val = running_corrects.double() / dataset_sizes[phase]

            # epoch_loss_vec.append(epoch_loss)
            # epoch_acc_vec.append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'train' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            new_data = {'epochs': [epoch],
            'trainlosses': [epoch_loss],
            'vallosses': [epoch_loss_val]}
            doc.add_next_tick_callback(partial(update, new_data))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # plot loss and accuracy
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(epoch_loss_vec, 'r')
    ax1.set_title('Epoch loss')
    ax2.plot(epoch_acc_vec, 'b')
    ax2.set_title('Accuracy')

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)






if __name__ == '__main__':

    # To convert data from PIL to tensor
    data_dir = '../../../chest_xray_pneumonia/'
    # data_dir = '../hymenoptera_data' # train model on generic images
    train_dir = data_dir + 'train'
    val_dir = data_dir + 'val'
    test_dir = data_dir + 'test'
    epochs = 5
    batch_size = 64

    # data_transforms = {
    #     'train': transforms.Compose([
    #         transforms.Resize((224, 224)),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    #     'val': transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    #     'test': transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    # }

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    test = torchvision.datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    testset = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


#####################################################################################
    source = ColumnDataSource(data={'epochs': [],
    'trainlosses': [],
    'vallosses': []}
    )

    # Add the plot to the current document
    plot = figure()
    plot.line(x= 'epochs', y ='trainlosses',
    color ='green', alpha = 0.8, legend ='Trainloss', line_width = 2,source = source)
    plot.line(x= 'epochs', y ='vallosses',
    color ='red', alpha = 0.8, legend ='Valloss', line_width = 2,
                        source = source)
    doc = curdoc()
    doc.add_root(plot)


    @gen.coroutine
    def update(new_data):
        source.stream(new_data)
############################################################################
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])


    os.environ['TORCH_HOME'] = 'models\\resnet' #setting the environment variable
    model_ft = models.resnet18(pretrained=True)

    ## fine tuning on fully connected layers
    # model_conv = torchvision.models.resnet18(pretrained=True)
    # for param in model_conv.parameters():
    #     param.requires_grad = False



    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft = model_ft.to(device)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


    # train model and save
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=epochs)


    #run the test function
    #test_model(model_ft,testset)


    #COPIED FROM INDIAN GUY's GITHUB
    correctHits = 0
    total = 0
    for batches in testset:
        data, output = batches
        data, output = data.to(device), output.to(device)
        prediction = model_ft(data)
        _, prediction = torch.max(prediction.data, 1)  # returns max as well as its index
        total += output.size(0)
        correctHits += (prediction == output).sum().item()

    print('Test accuracy = ' + str((correctHits / total) * 100))

    print('FINISHED')
    print()

    thread = Thread(target=train_model)
    thread.start()

    #COPIED:

    # img_name = "1.jpeg"  # change this to the name of your image file.def predict_image(image_path, model):
    # image = Image.open(image_path)
    # image_tensor = transforms(image)
    # image_tensor = image_tensor.unsqueeze(0)
    # image_tensor = image_tensor.to(device)
    # output = model(image_tensor)
    # index = output.argmax().item()
    # if index == 0:
    #     return "Cat"
    # elif index == 1:
    #     return "Dog"
    # else:
    #     returnpredict(img_name, model)






    # visualize model
    visualize_model(model_ft)

    # ## load a pretrained resnet18 model

    ## load a pretrained resnet18 model
    # model_conv = torchvision.models.resnet18(pretrained=True)
    # for param in model_conv.parameters():
    #     param.requires_grad = False
    #
    # # Parameters of newly constructed modules have requires_grad=True by default
    # num_ftrs = model_conv.fc.in_features
    # model_conv.fc = nn.Linear(num_ftrs, 2)
    #
    # model_conv = model_conv.to(device)
    #
    # criterion = nn.CrossEntropyLoss()
    #
    # # Observe that only parameters of final layer are being optimized as
    # # opposed to before.
    # optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
    #
    # # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    #
    # model_conv = train_model(model_conv, criterion, optimizer_conv,
    #                          exp_lr_scheduler, num_epochs=epochs)
    #                          exp_lr_scheduler, num_epochs=25)

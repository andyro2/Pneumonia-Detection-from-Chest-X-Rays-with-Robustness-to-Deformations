#from __future__ import print_function, division
import  torch
import  torchvision
import  time
import  os
import  copy

import  torch.nn            as nn
import  torch.optim         as optim
import  numpy               as np
import  matplotlib.pyplot   as plt

# from resnet_dcn import BasicBlock, Bottleneck, ResNet
from resnet_dcn_oeway import BasicBlock, Bottleneck, ResNet
# from alexnet import AlexNet
# from densenet import DenseNet
from    torch.optim import lr_scheduler
from    torchvision import datasets, models, transforms
from torch.utils import data



def imshow (inp, title=None):
    inp  = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    inp  = std*inp + mean
    inp  = np.clip(inp,0,1)
    plt.imshow(inp)

    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epoch_loss_vec, epoch_acc_vec = [], []
    loss_vec_val , acc_vec_val = [], []
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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                epoch_loss_vec.append(epoch_loss)
                epoch_acc_vec.append(epoch_acc)
            if phase == 'val':
                loss_vec_val.append(epoch_loss)
                acc_vec_val.append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # plot loss and accuracy
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(epoch_loss_vec, 'r', loss_vec_val, 'b')
    ax1.legend(['train', 'validation'])
    ax1.set_title('Epoch loss')
    ax1.grid()
    ax2.plot(epoch_acc_vec, 'r', acc_vec_val, 'b')
    ax2.legend(['train', 'validation'])
    ax2.set_title('Accuracy')
    ax2.grid()

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

def test_model(model_ft,testset):
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

def get_data(data_dir):

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
    # master = data.Dataset(data_dir)  # your "master" dataset
    # n = len(master);
    # n_train = int(0.6 * n)
    # n_val = int(0.2 * n)
    # n_test = n - n_train - n_val
    # train_set, val_set, test_set = data.random_split(master, (n_train, n_val, n_test))


    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    test = torchvision.datasets.ImageFolder(data_dir + 'test', transform=data_transforms['test'])
    testset = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)



    return dataloaders, testset, dataset_sizes, class_names


    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])




if __name__ == '__main__':

    # To convert data from PIL to tensor
    data_dir = '../../../Kaggle_Xray_pneoumonia/'
    # data_dir = '../hymenoptera_data' # train model on generic images
    epochs = 15
    batch_size = 64

    # get data
    dataloaders, testset, dataset_sizes, class_names = get_data(data_dir)

    #choose device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    #AlexNet
    # model_ft = AlexNet(num_classes=2)
    # ResNet-18
    model_ft = ResNet(BasicBlock,[2,2,2,2],num_classes=2)
    # # ResNet-50
    # model_ft = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=2)
    # # DenseNet121
    # model_ft = DenseNet(32, (6, 12, 24, 16), 64, num_classes=2)



    # model_ft.classifier = nn.Linear(1024, 2)


    model_ft = model_ft.to(device)

    # Loss criterion
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


    # train model and save
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=epochs)


    #run the test function
    test_model(model_ft,testset)



    print('FINISHED')
    print()





    # visualize model
    visualize_model(model_ft)
    print()
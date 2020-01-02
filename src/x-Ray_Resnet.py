import torch
import numpy as np
from torchvision import transforms, models, datasets
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt

class baseBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, input_planes, planes, stride=1, dim_change=None):
        super(baseBlock, self).__init__()
        # declare convolutional layers with batch norms
        self.conv1 = torch.nn.Conv2d(input_planes, planes, stride=stride, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, stride=1, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.dim_change = dim_change

    def forward(self, x):
        # Save the residue
        res = x
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))

        if self.dim_change is not None:
            res = self.dim_change(res)

        output += res
        output = F.relu(output)

        return output


class bottleNeck(torch.nn.Module):
    expansion = 4

    def __init__(self, input_planes, planes, stride=1, dim_change=None):
        super(bottleNeck, self).__init__()

        self.conv1 = torch.nn.Conv2d(input_planes, planes, kernel_size=1, stride=1)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv3 = torch.nn.Conv2d(planes, planes * self.expansion, kernel_size=1)
        self.bn3 = torch.nn.BatchNorm2d(planes * self.expansion)
        self.dim_change = dim_change

    def forward(self, x):
        res = x

        output = F.relu(self.bn1(self.conv1(x)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))

        if self.dim_change is not None:
            res = self.dim_change(res)

        output += res
        output = F.relu(output)
        return output


class ResNet(torch.nn.Module):
    def __init__(self, block, num_layers, classes=10):
        super(ResNet, self).__init__()
        # according to research paper:
        self.input_planes = 64
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=100, stride=4, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.layer1 = self._layer(block, 64, num_layers[0], stride=1)
        self.layer2 = self._layer(block, 128, num_layers[1], stride=2)
        self.layer3 = self._layer(block, 256, num_layers[2], stride=2)
        self.layer4 = self._layer(block, 512, num_layers[3], stride=2)
        self.averagePool = torch.nn.AvgPool2d(kernel_size=4, stride=1)
        self.fc = torch.nn.Linear(512 * block.expansion, classes)

    def _layer(self, block, planes, num_layers, stride=1):
        dim_change = None
        if stride != 1 or planes != self.input_planes * block.expansion:
            dim_change = torch.nn.Sequential(
                torch.nn.Conv2d(self.input_planes, planes * block.expansion, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(planes * block.expansion))
        netLayers = []
        netLayers.append(block(self.input_planes, planes, stride=stride, dim_change=dim_change))
        self.input_planes = planes * block.expansion
        for i in range(1, num_layers):
            netLayers.append(block(self.input_planes, planes))
            self.input_planes = planes * block.expansion

        return torch.nn.Sequential(*netLayers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



# def imshow(image):
#     if isinstance(image,torch.Tensor):
#         image = image.numpy().transpose((1, 2, 0))
#     else:
#         image = np.array(image).transpose((1, 2, 0))
#     # Un-normalize
#     mean = np.array([0.5])
#     std = np.array([0.229])
#     image = std * image + mean
#     image = np.clip(image, 0, 1)
#     # Plot
#     fig, ax = plt.subplots(1, 1, figsize=(15, 15))
#     plt.imshow(image)
#     ax.axis('off')

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





def test():
    # To convert data from PIL to tensor
    data_dir = './chest_xray_pneumonia/'
    train_dir = 'train'
    val_dir = 'val'
    test_dir = 'test'
    batch_size = 64

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        # transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load train and test set:
    train = torchvision.datasets.ImageFolder(data_dir + train_dir, transform=transform)
    trainset = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)

    val = torchvision.datasets.ImageFolder(data_dir + val_dir, transform=transform)
    valset = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True, drop_last=True)

    test = torchvision.datasets.ImageFolder(data_dir + test_dir, transform=transform)
    testset = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    iterable_set = iter(trainset)
    image, label = iterable_set.next()
    out = torchvision.utils.make_grid(image,nrow=8)
    #imshow(out)
    print(image.size())

    # ResNet-18
    # net = ResNet(baseBlock,[2,2,2,2],10)

    # ResNet-50
    net = ResNet(bottleNeck, [3, 4, 6, 3])
    net.to(device)
    costFunc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.02, momentum=0.9)






    for epoch in range(10):
        closs = 0
        for i, batch in enumerate(trainset, 0):
            data, output = batch
            data, output = data.to(device), output.to(device)
            prediction = net(data)
            loss = costFunc(prediction, output)
            closs = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print every 1000th time
            if i % 100 == 0:
                print('[%d  %d] loss: %.4f' % (epoch + 1, i + 1, closs / 1000))
                closs = 0
        correctHits = 0
        total = 0
        for batches in testset:
            data, output = batches
            data, output = data.to(device), output.to(device)
            prediction = net(data)
            _, prediction = torch.max(prediction.data, 1)  # returns max as well as its index
            total += output.size(0)
            correctHits += (prediction == output).sum().item()
        print('Accuracy on epoch ', epoch + 1, '= ', str((correctHits / total) * 100))

    correctHits = 0
    total = 0
    for batches in testset:
        data, output = batches
        data, output = data.to(device), output.to(device)
        prediction = net(data)
        _, prediction = torch.max(prediction.data, 1)  # returns max as well as its index
        total += output.size(0)
        correctHits += (prediction == output).sum().item()
    print('Accuracy = ' + str((correctHits / total) * 100))


if __name__ == '__main__':
    test()
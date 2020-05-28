import torch
import torch.nn as nn
# from dcn_oeway.torch_deform_conv.layers import ConvOffset2D # ModulatedDeformConv

# from .utils import load_state_dict_from_url


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

#
# class DCNAlexNet(nn.Module):
#
#     def __init__(self, num_classes=1000):
#         super(DCNAlexNet, self).__init__()
#
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
#         self.ReLU1 = nn.ReLU(inplace=True)
#         self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
#         self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
#         self.ReLU2 = nn.ReLU(inplace=True)
#         self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
#         self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
#         self.ReLU3 = nn.ReLU(inplace=True)
#         self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
#         self.ReLU4 = nn.ReLU(inplace=True)
#         self.conv5_offset = ConvOffset2D(256)
#         self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.ReLU5 = nn.ReLU(inplace=True)
#         self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
#
#         self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_classes),
#         )
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.ReLU1(out)
#         out = self.pool1(out)
#         out = self.conv2(out)
#         out = self.ReLU2(out)
#         out = self.pool2(out)
#         out = self.conv3(out)
#         out = self.ReLU3(out)
#         out = self.conv4(out)
#         out = self.ReLU4(out)
#         out = self.conv5_offset(out)
#         out = self.conv5(out)
#         x = self.pool3(out)
#
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x
#


def alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls['alexnet'],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    # return model

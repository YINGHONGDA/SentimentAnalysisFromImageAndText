import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from torchvision.models.vgg import model_urls
import time
import random
from VGG import VGG
from PIL import ImageFile
from ResNet import Resnet
from BottleNeck import Bottleneck
from collections import namedtuple

ImageFile.LOAD_TRUNCATED_IMAGES = True

pretrianed_size = 224
pretrianed_mean = [0.485, 0.456, 0.406]
pretrianed_stds = [0.229, 0.224, 0.225]

VALID_RATIO = 0.9
N_IMAGE = 25

BATCH_SIZE = 32
START_LR = 1e-7
OUTPUT_DIM = 8

FOUND_LR = 1e-3
EPOCH = 10

print(torch.cuda.is_available())

# set radom seed
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# LOAD IMAGE DATA
train_dic = '/home/yinghongda/DataSet/ImageDataset-update/train'
test_dic = '/home/yinghongda/DataSet/ImageDataset-update/test'

train_transforms = transforms.Compose([
    transforms.Resize(pretrianed_size),
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomCrop(pretrianed_size, padding=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=pretrianed_mean,
                         std=pretrianed_stds)
])

test_transforms = transforms.Compose([
    transforms.Resize(pretrianed_size),
    transforms.CenterCrop(pretrianed_size),
    # transforms.RandomCrop(pretrianed_size, padding=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=pretrianed_mean,
                         std=pretrianed_stds)
])

train_data = datasets.ImageFolder(train_dic, transform=train_transforms)
test_data = datasets.ImageFolder(test_dic, transform=test_transforms)

train_iterator = data.DataLoader(
    train_data,
    shuffle=True,
    batch_size=BATCH_SIZE
)
test_iterator = data.DataLoader(
    test_data,
    batch_size=BATCH_SIZE
)


# LOAD VGG_LAYERS
def get_vgg_layers(config, batch_norm):
    layers = []
    in_channels = 3
    for c in config:
        assert c == 'M' or isinstance(c, int)
        if c == 'M':
            layers += [nn.MaxPool2d(kernel_size=2)]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = c
    return nn.Sequential(*layers)


vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
vgg19_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_block', 'channels'])
resnet50_config = ResNetConfig(
    block=Bottleneck,
    n_block=[3, 4, 6, 3],
    channels=[64, 128, 256, 512]
)

#vgg_layers = get_vgg_layers(vgg19_config, batch_norm=True)

#model = VGG(vgg_layers, OUTPUT_DIM)



# LOADING PRETRAIN MODEL
# model_urls['vgg16_bn'] = model_urls['vgg16_bn'].replace('https://','http://')
#pretrain_model = models.vgg19_bn(pretrained=True)

pretrain_model = models.resnet50(pretrained=True)
# CHANGE OUT_DIM 1000->119
IN_FEATURES = pretrain_model.fc.in_features
final_fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
pretrain_model.fc = final_fc

model = Resnet(resnet50_config,OUTPUT_DIM)
model.load_state_dict(pretrain_model.state_dict())

# LOSSFUNCTION AND OPTIMIZER
optimizer = optim.Adam(model.parameters(), lr=START_LR)
loss_fun = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
loss_fun.to(device)

# DISCRIMINATIVE FINE_TUNING
params = [
    {'params': model.conv1.parameters(), 'lr': FOUND_LR/10},
    {'params': model.bn1.parameters(),'lr': FOUND_LR/10},
    {'params': model.layer1.parameters(), 'lr': FOUND_LR/8},
    {'params': model.layer2.parameters(), 'lr': FOUND_LR/6},
    {'params': model.layer3.parameters(), 'lr': FOUND_LR/4},
    {'params': model.layer4.parameters(), 'lr': FOUND_LR/2}
]

optimizer = optim.Adam(params, lr=START_LR)


# CALCULATE ACCURACY FUNCTION
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


# TRAIN FUNCTION
def train(model, iterator, optimizer, loss_fun, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_pred, _ = model(x)

        loss = loss_fun(y_pred, y)
        acc = calculate_accuracy(y_pred, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, loss_fun, device):
    epoch_acc = 0
    epoch_loss = 0

    model.eval()

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)
            loss = loss_fun(y_pred, y)
            acc = calculate_accuracy(y_pred, y)

            epoch_acc += acc.item()
            epoch_loss += loss.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_sec = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_sec


best_valid = float('inf')

for epoch in range(EPOCH):
    start_time = time.monotonic()
    train_loss, train_acc = train(model, train_iterator, optimizer, loss_fun, device)
    test_loss, test_acc = evaluate(model, test_iterator, loss_fun, device)

    if test_loss < best_valid:
        best_valid = test_loss
        torch.save(model.state_dict(), 'ResNet_model.pt')

    end_time = time.monotonic()

    epoch_mins, epoch_sec = epoch_time(start_time, end_time)

    print(f'Epoch:{epoch + 1:02}|Epoch Time:{epoch_mins}m{epoch_sec}s')
    print(f'\tTrain Loss:{train_loss:.3f}|Train Acc:{train_acc * 100:.2f}%')
    print(f'\tTest Loss:{test_loss:.3f}|Test Acc:{test_acc * 100:.2f}%')

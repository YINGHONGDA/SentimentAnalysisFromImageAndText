import torch
from PIL import Image
from torchvision import transforms
from VGG import VGG
from ResNet import Resnet
import torch.nn as nn
#import ImageClassification
import torchvision.datasets as datasets
#from ImageClassification import get_vgg_layers
from BottleNeck import Bottleneck
from collections import namedtuple
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

pretrianed_size = 224
pretrianed_mean = [0.485, 0.456, 0.406]
pretrianed_stds = [0.229, 0.224, 0.225]
OUTPUT_DIM = 8
BATCH_SIZE = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dic = '/home/yinghongda/DataSet/ImageDataset-update/train'
train_data = datasets.ImageFolder(train_dic)
class_name = train_data.class_to_idx
class_name = {v: k for k, v in class_name.items()}

test_transforms = transforms.Compose([
    transforms.Resize(pretrianed_size),
    transforms.CenterCrop(pretrianed_size),
    # transforms.RandomCrop(pretrianed_size, padding=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=pretrianed_mean,
                         std=pretrianed_stds)
])


ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_block', 'channels'])
resnet50_config = ResNetConfig(
    block=Bottleneck,
    n_block=[3, 4, 6, 3],
    channels=[64, 128, 256, 512]
)


model = Resnet(resnet50_config,OUTPUT_DIM)
model.load_state_dict(torch.load('ResNet_model.pt'))
model.to(device)



# def get_vgg_layers(config, batch_norm):
#     layers = []
#     in_channels = 3
#     for c in config:
#         assert c == 'M' or isinstance(c, int)
#         if c == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2)]
#         else:
#             conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = c
#     return nn.Sequential(*layers)
#
# vgg19_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
#                 'M']
# vgg_layers = get_vgg_layers(vgg19_config, batch_norm=True)

def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min = image_min, max = image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image

def predict(img_path):
    model = Resnet(resnet50_config,OUTPUT_DIM)
    model.load_state_dict(torch.load('ResNet_model.pt'))
    model = model.to(device)
    torch.no_grad()
    img = Image.open(img_path)
    img = test_transforms(img).unsqueeze_(0)
    img_ = img.to(device)
    pre,_ = model(img_)
    x,y = torch.max(pre,1)
    y = y[0].item()

    return class_name[y]
    #_,pre = torch.max(output,1)
    #print("this picture maybe:",pre[0])


def plot_confusion_matrix(labels, pred_labels, classes):
    fig = plt.figure(figsize=(10, 10));
    ax = fig.add_subplot(1, 1, 1);
    cm = confusion_matrix(labels, pred_labels);
    cm = ConfusionMatrixDisplay(cm, display_labels=classes);
    cm.plot(values_format='d', cmap='Blues', ax=ax)
    fig.delaxes(fig.axes[1])  # delete colorbar
    plt.xticks(rotation=90)
    plt.xlabel('Predicted Label', fontsize=10)
    plt.ylabel('True Label', fontsize=10)
    plt.show()


def plot_most_incorrect(incorrect, classes, n_images, normalize=True):
    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(25, 20))

    for i in range(rows * cols):

        ax = fig.add_subplot(rows, cols, i + 1)

        image, true_label, probs = incorrect[i]
        image = image.permute(1, 2, 0)
        true_prob = probs[true_label]
        incorrect_prob, incorrect_label = torch.max(probs, dim=0)
        true_class = classes[true_label]
        incorrect_class = classes[incorrect_label]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.cpu().numpy())
        ax.set_title(f'true label: {true_class} ({true_prob:.3f})\n' \
                     f'pred label: {incorrect_class} ({incorrect_prob:.3f})')
        ax.axis('off')

    fig.subplots_adjust(hspace=0.4)
    plt.show()



def get_predictions(model, iterator):

    model.eval()

    images = []
    labels = []
    probs = []

    with torch.no_grad():

        for (x, y) in iterator:

            x = x.to(device)

            y_pred, _ = model(x)

            y_prob = F.softmax(y_pred, dim = -1)
            top_pred = y_prob.argmax(1, keepdim = True)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim = 0)
    labels = torch.cat(labels, dim = 0)
    probs = torch.cat(probs, dim = 0)

    return images, labels, probs
test_dic  = '/home/yinghongda/DataSet/ImageDataset-update/test'
image_dic = '/home/yinghongda/DataSet/predict/beautiful_flower.jpg'


if __name__ == '__main__':
    predicate_imag = datasets.ImageFolder(test_dic,transform=test_transforms)
    prediction_iterator = data.DataLoader(predicate_imag,batch_size=BATCH_SIZE)
    images, labels, probs = get_predictions(model,prediction_iterator)
    pred_labels = torch.argmax(probs, 1)

    plot_confusion_matrix(labels, pred_labels, predicate_imag.classes)
    #
    # corrects = torch.eq(labels, pred_labels)
    # incorrect_examples = []
    #
    # for image, label, prob, correct in zip(images, labels, probs, corrects):
    #     if not correct:
    #         incorrect_examples.append((image, label, prob))
    #
    # incorrect_examples.sort(reverse=True, key=lambda x: torch.max(x[2], dim=0).values)
    # N_IMAGES = 36
    #plot_most_incorrect(incorrect_examples, predicate_imag.classes, N_IMAGES)



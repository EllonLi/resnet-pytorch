import torch
from torch import nn, optim
import torchvision
from torchvision import transforms
import argparse
import os
import time
#from utils import *
from resnet_model import *
from PIL import Image
import torch.nn.functional as F
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

lr = [0.01]

my_open = open('test-resnetreer18.txt', 'a')
# save model name


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser("""medical image classification!""")
    parser.add_argument('--path', type=str, default='data',
                        help="""image dir path default: 'dataset'.""")
    parser.add_argument('--workers', type=int, default=8, metavar='N',
                        help="""number of data loading workers, default:4.""")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="""Batch_size default:256.""")
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--num_classes', type=int, default=6,
                        help="""num classes""")
    parser.add_argument('--model_path', type=str, default='model/model_28.pkl',
                        help="""Save model path""")
    parser.add_argument('--pretrained_model', type=bool, default=False,
                        help="""if use the pretrained imagenet model""")

    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
    ])

    pathdir="./data/val1/"

    # Load data
    model=torch.load(args.model_path)

    my_open.write('\n*****************{}*******************\n'.format(args.model_path))
    model.eval()
    correct_prediction = 0.
    total = [0, 0, 0, 0, 0, 0]
    label_predicted = [0, 0, 0, 0, 0, 0]
    for file in os.listdir(pathdir):
        if file.endswith('.jpg'):
            print('current file: %s' %file)
            fopen=Image.open(os.path.join(pathdir,file)).convert('RGB')
            transform = transforms.Compose([
                            transforms.Resize(224),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
                            ])
            image=transform(fopen)
            image=image.unsqueeze(0)
            image = image.to(device)
            #labels = labels.to(device)
            torch.no_grad()
            outputs = model(image)
        # equal prediction and acc
            _, predicted = torch.max(outputs.data, 1)
            predicted_numpy = predicted.cpu().numpy()
            
            # val_loader total
            # total += labels.size(0)
            # add correct
            # correct_prediction += (predicted == labels).sum().item()
            print(predicted_numpy[0])
            label_predicted[predicted_numpy[0]] += 1
    print(label_predicted)
    strlog="result: %s"%(label_predicted)
    my_open.write(strlog)
my_open.close()

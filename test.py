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

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

lr = [0.01]

my_open = open('test-resnet18.txt', 'a')
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
        #transforms.Resize(224),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
    ])


    # Load data
    test_datasets = torchvision.datasets.ImageFolder(root=os.path.join(args.path, 'val'),
                                                      transform=transform)

    test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)

    model=torch.load(args.model_path)
    
    my_open.write('\n********************{}***********************\n'.format(args.model_path))
    model.eval()
    correct_prediction = 0.
    total = [0, 0, 0, 0, 0, 0]
    label_predicted = [[0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0]]
    for images, labels in test_loader:
        # to GPU
        images = images.to(device)
        labels = labels.to(device)
        # print prediction
        outputs = model(images)
        # equal prediction and acc
        _, predicted = torch.max(outputs.data, 1)

        labels_numpy = labels.cpu().numpy()
        predicted_numpy = predicted.cpu().numpy()

        for count in range(len(labels_numpy)):
            total[labels_numpy[count]] += 1
            if labels_numpy[count] == 0:
                label_predicted[0][predicted_numpy[count]] += 1
            if labels_numpy[count] == 1:
                label_predicted[1][predicted_numpy[count]] += 1
            if labels_numpy[count] == 2:
                label_predicted[2][predicted_numpy[count]] += 1
            if labels_numpy[count] == 3:
                label_predicted[3][predicted_numpy[count]] += 1
            if labels_numpy[count] == 4:
                label_predicted[4][predicted_numpy[count]] += 1
            if labels_numpy[count] == 5:
                label_predicted[5][predicted_numpy[count]] += 1
            # val_loader total
            # total += labels.size(0)
            # add correct
            # correct_prediction += (predicted == labels).sum().item()

    print("total:" + str(total[:]))
    print("label[0]: " + str(label_predicted[0][:]))
    print("label[1]: " + str(label_predicted[1][:]))
    print("label[2]: " + str(label_predicted[2][:]))
    print("label[3]: " + str(label_predicted[3][:]))
    print("label[4]: " + str(label_predicted[4][:]))
    print("label[5]: " + str(label_predicted[5][:]))
    print("ACC_0: " + str(label_predicted[0][0] / total[0]))
    print("ACC_1: " + str(label_predicted[1][1] / total[1]))
    print("ACC_2: " + str(label_predicted[2][2] / total[2]))
    print("ACC_3: " + str(label_predicted[3][3] / total[3]))
    print("ACC_4: " + str(label_predicted[4][4] / total[4]))
    print("ACC_5: " + str(label_predicted[5][5] / total[5]))
        # my_open.write("ACC:  " + str(correct_prediction/total) + '\n')
    strlog="total:" + str(total[:])+ '\n'
    my_open.write(strlog)
    strlog="label[0]: " + str(label_predicted[0][:])+ '\n'
    my_open.write(strlog)
    strlog="label[1]: " + str(label_predicted[1][:])+ '\n'
    my_open.write(strlog)
    strlog="label[2]: " + str(label_predicted[2][:])+ '\n'
    my_open.write(strlog)
    strlog="label[3]: " + str(label_predicted[3][:])+ '\n'
    my_open.write(strlog)
    strlog="label[4]: " + str(label_predicted[4][:])+ '\n'
    my_open.write(strlog)
    strlog="label[5]: " + str(label_predicted[5][:])+ '\n'
    my_open.write(strlog)
    strlog="ACC_0: " + str(label_predicted[0][0] / total[0])+ '\n'
    my_open.write(strlog)
    strlog="ACC_1: " + str(label_predicted[1][1] / total[1])+ '\n'
    my_open.write(strlog)
    strlog="ACC_2: " + str(label_predicted[2][2] / total[2])+ '\n'
    my_open.write(strlog)
    strlog="ACC_3: " + str(label_predicted[3][3] / total[3])+ '\n'
    my_open.write(strlog)
    strlog="ACC_4: " + str(label_predicted[4][4] / total[4])+ '\n'
    my_open.write(strlog)
    strlog="ACC_5: " + str(label_predicted[5][5] / total[5])+ '\n'
    my_open.write(strlog)

my_open.close()

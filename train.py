#coding=utf-8
import torch
from torch import nn, optim
import torchvision
from torchvision import transforms
import argparse
import os
import time
#from utils import *
from resnet_model_18_2 import *
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

lr = [0.001]

my_open = open('loss-resnet18-2-1.txt', 'a')
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
    parser.add_argument('--batch_size', type=int, default=16,
                        help="""Batch_size default:256.""")
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--num_classes', type=int, default=6,
                        help="""num classes""")
    parser.add_argument('--model_path', type=str, default='model18-3-1',
                        help="""Save model path""")
    parser.add_argument('--pretrained_model', type=bool, default=False,
                        help="""if use the pretrained imagenet model""")

    args = parser.parse_args()

    # Create model
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    transform = transforms.Compose([
        transforms.Resize(230),  # 将图像转化为32 * 32
        # transforms.RandomHorizontalFlip(p=0.75),  # 有0.75的几率随机旋转
        transforms.RandomCrop(230),  # 从图像中裁剪一个24 * 24的
        transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
    ])

    # Load data
    train_datasets = torchvision.datasets.ImageFolder(root=os.path.join(args.path, 'train'),
                                                      transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_datasets,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)

    # Load data
    test_datasets = torchvision.datasets.ImageFolder(root=os.path.join(args.path, 'val'),
                                                      transform=transform)

    test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                               batch_size=1,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)


    model = resnet18(pretrained=False).to(device)
    model.fc = nn.Linear(512, args.num_classes).to(device)
    cast = nn.CrossEntropyLoss().to(device)

    my_open.write('\n***************************************************\n')

    for i in range(len(lr)):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr[i], momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        for epoches in range(1, 61):
            print(epoches)

            model.train()
            temp=0
            for images, labels in train_loader:
                start = time.time()
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = cast(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                temp+=1
                #print("train batch-size:",temp)
                end = time.time()
                    # 打印出来时间信息
            print("Epoch:", epoches, "Loss:", loss.item(), "Time:", (end-start) * (30-epoches))
            my_open.write("Epoch: " + str(epoches) + "   Loss: " + str(loss.item()) + '\n')
            torch.save(model,os.path.join(args.model_path,'model_{}.pkl'.format(epoches)))
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

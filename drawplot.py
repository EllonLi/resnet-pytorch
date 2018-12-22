import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
txt='loss-resnet18.txt'
def draw_loss():
    x_epoch=[]
    y_loss=[]
    with open(txt,'r') as fi:
        lines=fi.readlines()
        for line in lines:
            if 'Epoch:' in line:
                #print (line)
                tempstr=line.split('   ')
                #print(tempstr)
                x_epoch.append(int(tempstr[0].split(' ')[1]))
                y_loss.append(round(float(tempstr[1].split(' ')[1]),4))
        print(x_epoch)
        print(y_loss)
    plt.figure()
    plt.plot(x_epoch,y_loss,color='red')
    #设置x轴y轴标题
    plt.xlabel(u'epoch',fontproperties='SimHei',fontsize=14)
    plt.ylabel(u'loss',fontproperties='SimHei',fontsize=14)
    #设置x轴刻度
    plt.xticks(np.arange(31))
    
    plt.savefig(str(txt.split('.')[0]+'_loss.jpg'))
    plt.show()
    
def draw_acc():
    x_epoch=[]
    y0=[]
    y1=[]
    y2=[]
    y3=[]
    y4=[]
    y5=[]
    count=0
    with open(txt,'r') as fi:
        lines=fi.readlines()
        for line in lines:
            if 'Epoch:' in line:
                #print (line)
                tempstr=line.split('   ')
                #print(tempstr)
                x_epoch.append(int(tempstr[0].split(' ')[1]))
            if 'ACC' in line:
                temp=line.split(': ')
                if temp[0]=='ACC_0':
                    y0.append(float(temp[1]))
                elif temp[0]=='ACC_1':
                    y1.append(float(temp[1]))
                elif temp[0]=='ACC_2':
                    y2.append(float(temp[1]))
                elif temp[0]=='ACC_3':
                    y3.append(float(temp[1]))
                elif temp[0]=='ACC_4':
                    y4.append(float(temp[1]))
                elif temp[0]=='ACC_5':
                    y5.append(float(temp[1]))
        #print(y0)
    plt.figure()
    plt.plot(x_epoch,y0,color='r',label='atrophy')
    plt.plot(x_epoch,y1,color='b',label='cancer')
    plt.plot(x_epoch,y2,color='g',label='gist')
    plt.plot(x_epoch,y3,color='m',label='negative')
    plt.plot(x_epoch,y4,color='y',label='ulcer')
    plt.plot(x_epoch,y5,color='k',label='polyp')
    #设置x轴y轴标题
    plt.xlabel(u'epoch',fontsize=14)
    plt.ylabel(u'acc',fontsize=14)
    
    plt.legend(loc='lower right')
    plt.savefig(str(txt.split('.')[0]+'_acc.jpg'))
    plt.show()

if __name__ == '__main__':
    draw_loss()
    draw_acc()

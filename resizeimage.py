#-*-coding:utf-8-*-
"""
Created on 2018 11 19

@author: Ellon
"""
import cv2
import  os

def resizeimg():
    input_path='/home/hxjy/lj/resnet-pytorch/data/val/ulcer-crop'
    output_path='/home/hxjy/lj/resnet-pytorch/data/val/ulcer-crop'
    filenames = os.listdir(input_path)
    for filename in filenames:
        if filename.endswith('.jpg'):
            print(filename)
            filepath = os.path.join(input_path, filename)
            img=cv2.imread(filepath)
            if img is None:
                os.remove(input_path+'/'+filename)
            else:
                img=cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
                filepath1=os.path.join(output_path,filename)
                cv2.imwrite(filepath1,img)




if __name__ == "__main__":
    print('start.....')
    resizeimg()
    print ('finish')

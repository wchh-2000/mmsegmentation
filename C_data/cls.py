from PIL import Image
import numpy as np
from tqdm import tqdm 
from matplotlib import pyplot as plt
import os
def count(dir):
    name=os.listdir(dir)
    label_pths=[dir+n for n in name]
    cnt=[0]*9
    for label_pth in tqdm(label_pths):
        label=np.array(Image.open(label_pth))
        for i in range(9):
            cnt[i]+=np.sum(label==i)
    cnt1=[0]*9
    n=sum(cnt)
    for i in range(9):
        cnt1[i]=cnt[i]/n #归一化
    print(cnt1)
    return cnt1
'''
train:
[0.028405396034465825, 0.21274933290295753, 0.0500320258311337, 0.2186900694707834, \
0.0548619401103204, 0.13375789262209206, 0.24163253658179296, 0.044734245550878016, 0.015136560895576131]
'''
classes=['background', 'water','transport', 'building','agricultural',\
    'grass', 'forest','barren','others']

dir_train="../chusai_release/train/labels_9/"
dir_test="../results/"
cnt1,cnt2=count(dir_train),count(dir_test)
y=np.arange(len(classes))
width=0.25
plt.barh(y-width/2,cnt1,label="train",height=width)
plt.barh(y+width/2,cnt2,label="test",height=width)
plt.legend()
plt.yticks(y, labels=classes)
plt.title('classes proportion')
plt.savefig("train&test_classes_cnt.png")
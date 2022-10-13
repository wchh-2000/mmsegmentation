from PIL import Image
import numpy as np
from tqdm import tqdm 
from matplotlib import pyplot as plt
import os
def count(dir,test=False):
    n=18
    name=os.listdir(dir)
    label_pths=[dir+n for n in name]
    cnt=[0]*n
    for label_pth in tqdm(label_pths):
        label=np.array(Image.open(label_pth))
        if test:
            label=label%100
        for i in range(n):
            cnt[i]+=np.sum(label==i)
    cnt1=[0]*n
    N=sum(cnt)
    for i in range(n):
        cnt1[i]=cnt[i]/N #归一化
    print(cnt1)
    print(sum(cnt1))
    return cnt1
'''
train:
初赛：
[0.028405396034465825, 0.21274933290295753, 0.0500320258311337, 0.2186900694707834, \
0.0548619401103204, 0.13375789262209206, 0.24163253658179296, 0.044734245550878016, 0.015136560895576131]
复赛：
[0.028405396034465825, 0.21274933290295753, 0.049147634387657804, 0.2186900694707834, \
0.000473727496839627, 0.00041066394663626864, 1.866150751784596e-05, 0.006398515104915685, 0.002896648144363627, 0.04811980459326869, 0.006742135517051698, 0.07285460710835205, 0.06090328551374001, 0.20532427278293222, 0.03630826379886074, 0.00044290164247428937, 0.04429134390840373, 0.005822736138778973]

测试：
[0.026848597273136102, 0.042382089176856846, 0.01617045149112666, 0.06154481542802101, 0.0, 0.0, 0.0, 0.001029473522824232, 0.0006478568625299016, 0.10393093691286025, 0.0018792805701915605, 0.03320280416368682, 0.005117241549294266, 0.645524277447608, 0.046005214614905365, 0.0003787663318888842, 0.014886844919577626, 0.00045134973549249983]

'''
classes=['Background','Waters', 'Road', 'Construction', 'Airport', 'Railway Station', 'Photovoltaic panels', 'Parking Lot', 'Playground',
           'Farmland', 'Greenhouse', 'Grass', 'Artificial grass', 'Forest', 'Artificial forest', 'Bare soil', 'Artificial bare soil', 'Other']

dir_train="/data/fusai_release/train/labels_18/"

dir_test="../C_run/results/"

# cnt1=count(dir_train)
cnt1=[0.028405396034465825, 0.21274933290295753, 0.049147634387657804, 0.2186900694707834, \
0.000473727496839627, 0.00041066394663626864, 1.866150751784596e-05, 0.006398515104915685, \
0.002896648144363627, 0.04811980459326869, 0.006742135517051698, 0.07285460710835205, \
0.06090328551374001, 0.20532427278293222, 0.03630826379886074, 0.00044290164247428937, \
    0.04429134390840373, 0.005822736138778973]
cnt2=count(dir_test,test=True)
y=np.arange(len(classes))
width=0.4
plt.figure(figsize=(12,6))
plt.barh(y-width/2,cnt1,label="train",height=width)
plt.barh(y+width/2,cnt2,label="test",height=width)
plt.legend()
plt.grid(True)
plt.yticks(y, labels=classes)
plt.title('classes proportion')
plt.savefig("train&test_classes_fu.png")
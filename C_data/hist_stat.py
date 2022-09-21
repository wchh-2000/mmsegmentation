import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import mmcv
dir="../chusai_release/train/images/"
names=list(range(10778))
def get_width(h):#list 各亮度频数
    if h[0]>200:
        return 0
    i=1
    s=0#start
    while i<256:
        if h[i]>100 and s==0:#开始
            s=i
        if h[i]<100 and s!=0:#结束
            return i-s
        i+=1
    return i-s
def process1(name):
    img_pth=dir+name
    img=Image.open(img_pth)
    r,g,b=img.convert("RGB").split()
    H=[]
    Vm=[]
    width=[]
    for c in [r,g,b]:
        h=c.histogram()
        Vm.append(h.index(max(h)))
        width.append(get_width(h))
    # H=np.array(H)
    Vmax_mean=sum(Vm)/3
    width=sum(width)/3
    # Vmax_std=np.std(np.array(Vm))
    # if Vmax_mean<5:
    #     print(Vmax_mean,name)
    r=(Vmax_mean,width)#np.std(H)
    return r
def process(name):
    img_pth=dir+str(name)+'.tif'
    img=Image.open(img_pth)
    r,g,b=img.convert("RGB").split()
    H=[]
    Vm=[]
    width=[]
    h=b.histogram()
    Vm=h.index(max(h))
    
    width=get_width(h)
    r=[Vm,width]#,name]
    return r
result = mmcv.track_parallel_progress(process, names, 16)
# print([r[2] for r in result[:20]])
np.save("hist.npy",np.array(result,dtype=np.uint8))
# Vmax=[r[0] for r in result]
# width=[r[1] for r in result]
# plt.scatter(Vmax,width,s=1)
# plt.xlabel("Vmax")
# plt.ylabel("width")
# plt.show()
# plt.savefig("histB.png")
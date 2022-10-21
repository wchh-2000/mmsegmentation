import numpy as np
import albumentations as A
from PIL import Image
from matplotlib import pyplot as plt
import time
from numpy.fft import fft2,fftshift
from math import sqrt,pow
from functools import wraps
MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}
def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)
def clipped(func):
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        dtype = img.dtype
        maxval = MAX_VALUES_BY_DTYPE.get(dtype, 1.0)
        return clip(func(img, *args, **kwargs), dtype, maxval)

    return wrapped_function
def get_filter(D0=0.05,n=3):#获取巴特沃斯低、高通滤波器频谱
    #D0:截止频率使得lpf(u,v)=0.5 
    #n:滤波器阶数
    shape=512
    lpf=np.zeros((shape,shape))#一通道的尺寸
    hpf=np.zeros((shape,shape))
    M,N=lpf.shape
    for u in range(M):
        for v in range(N):
            D=sqrt((u-M/2)**2+(v-N/2)**2)
            lpf[u,v]=1/(1+pow(D/D0,2*n))
            hpf[u,v]=1-lpf[u,v]
    return lpf,hpf
@clipped    
def fda_butterworth(img: np.ndarray, target_img: np.ndarray,D0:float) -> np.ndarray:
    """巴特沃斯滤波器
    Args:
        img:  source image
        target_img:  target image for domain adaptation
    """
    img=img.transpose(2,1,0)
    target_img=target_img.transpose(2,1,0)
    # get fft of both source and target
    fft_src = np.fft.fft2(img.astype(np.float32))
    fft_trg = np.fft.fft2(target_img.astype(np.float32))

    # extract amplitude and phase of both fft-s
    amplitude_src, phase_src = np.abs(fft_src), np.angle(fft_src)
    amplitude_trg = np.abs(fft_trg)

    # mutate the amplitude part of source with target
    amplitude_src = np.fft.fftshift(amplitude_src)
    amplitude_trg = np.fft.fftshift(amplitude_trg)

    lpf,hpf=get_filter(D0=D0,n=3)
    amplitude_src=amplitude_src*hpf+amplitude_trg*lpf

    amplitude_src = np.fft.ifftshift(amplitude_src)

    # get mutated image
    src_image_transformed = np.fft.ifft2(amplitude_src * np.exp(1j * phase_src))
    src_image_transformed = np.real(src_image_transformed)
    src_image_transformed=src_image_transformed.transpose(2,1,0)
    return src_image_transformed
@clipped
def fourier_domain_adaptation(img: np.ndarray, target_img: np.ndarray,
    beta: float,D0=0.1,Butterworth=False) -> np.ndarray:
    """
    Fourier Domain Adaptation from https://github.com/YanchaoYang/FDA

    Args:
        img:  source image
        target_img:  target image for domain adaptation
        beta: coefficient from source paper

    Returns:
        transformed image

    """

    img = np.squeeze(img)
    target_img = np.squeeze(target_img)

    if target_img.shape != img.shape:
        raise ValueError(
            "The source and target images must have the same shape,"
            " but got {} and {} respectively.".format(img.shape, target_img.shape)
        )

    # get fft of both source and target
    fft_src = np.fft.fft2(img.astype(np.float32), axes=(0, 1))
    fft_trg = np.fft.fft2(target_img.astype(np.float32), axes=(0, 1))

    # extract amplitude and phase of both fft-s
    amplitude_src, phase_src = np.abs(fft_src), np.angle(fft_src)
    amplitude_trg = np.abs(fft_trg)

    # mutate the amplitude part of source with target
    amplitude_src = np.fft.fftshift(amplitude_src, axes=(0, 1))
    amplitude_trg = np.fft.fftshift(amplitude_trg, axes=(0, 1))

    #频谱矩形窗替换：
    if not Butterworth:
        height, width = amplitude_src.shape[:2]
        border = np.floor(min(height, width) * beta).astype(int)
        center_y, center_x = np.floor([height / 2.0, width / 2.0]).astype(int)

        y1, y2 = center_y - border, center_y + border + 1
        x1, x2 = center_x - border, center_x + border + 1

        amplitude_src[y1:y2, x1:x2] = amplitude_trg[y1:y2, x1:x2]    
    else:#巴特沃斯滤波器：
        lpf,hpf=get_filter(D0=D0,n=3)
        amplitude_src=amplitude_src.transpose(2,1,0)
        amplitude_trg=amplitude_trg.transpose(2,1,0)
        amplitude_src=amplitude_src*hpf+amplitude_trg*lpf
        amplitude_src=amplitude_src.transpose(2,1,0)

    amplitude_src = np.fft.ifftshift(amplitude_src, axes=(0, 1))

    # get mutated image
    src_image_transformed = np.fft.ifft2(amplitude_src * np.exp(1j * phase_src), axes=(0, 1))
    src_image_transformed = np.real(src_image_transformed)

    return src_image_transformed

dir="/data/fusai_release/train/images/"
src_id=10074#70
trg_id=4
image=Image.open(dir+str(src_id)+".tif")
target=Image.open("/data/mmseg/C_data/shift4.png")#dir+str(trg_id)+".tif"
# aug = A.Compose([A.FDA([dir+"4.tif",dir+"3.tif"], p=1,beta_limit=0.001)])
# #随机从文件列表中选取作为target
# result = aug(image=np.array(image))['image']
s=time.time()
result=fda_butterworth(np.array(image),np.array(target),D0=0.1)
# result=fourier_domain_adaptation(np.array(image),np.array(target),beta=0.005)
print(time.time()-s)#0.2s 巴特沃斯0.86s
#替换幅度谱中心2beta长的正方形
result=Image.fromarray(result)

x=range(256)
plt.subplots(3,3,figsize=(9,9))
def normalize(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
title=['original '+str(src_id),'target' +str(trg_id),'result']
for i,pack in enumerate(zip([image,target,result],title)):
    img,t=pack
    r,g,b=img.convert("RGB").split()
    plt.subplot(3,3,3*i+1)
    plt.imshow(img)
    plt.title(t)
    plt.subplot(3,3,3*i+2)
    for channel,color in zip([r,g,b],['r','g','b']):
        plt.plot(x,channel.histogram(),color=color)
    plt.subplot(3,3,3*i+3)
    #画频谱：
    img=np.array(img)
    F=np.log(abs(fftshift(fft2(img)))+1)#加一防止log(0)
    # print(F.min(),F.max())
    plt.imshow(normalize(F))#float 归一化

plt.savefig("/data/mmseg/C_data/fda1.png") 
import numpy as np
import albumentations as A
from PIL import Image
import mmcv
import glob
from numpy.fft import fft2,fftshift,ifft2,ifftshift
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
lpf,hpf=get_filter(D0=0.1,n=3)
img_paths = glob.glob('/data/fusai_release/test/images/*.tif')
# trg_pth="/data/fusai_release/train/images/4.tif"
target=np.array(Image.open("/data/mmseg/C_data/shift4.png"))
# mean=[117.60938644,115.98219299,118.33253479],#rgb
# std=[20.8069714,20.73299022,24.21787756],
target=target.transpose(2,1,0)
amplitude_trg = fftshift(np.abs(fft2(target.astype(np.float32))))
@clipped    
def fda_butterworth(img: np.ndarray) -> np.ndarray:
    """巴特沃斯滤波器
    Args:
        img:  source image
        target_img:  target image for domain adaptation
        """
    img=img.transpose(2,1,0)
    # get fft of both source and target
    fft_src = fft2(img.astype(np.float32))

    # extract amplitude and phase of both fft-s
    amplitude_src, phase_src = np.abs(fft_src), np.angle(fft_src)
    # mutate the amplitude part of source with target
    amplitude_src = fftshift(amplitude_src)

    amplitude_src=amplitude_src*hpf+amplitude_trg*lpf
    amplitude_src = ifftshift(amplitude_src)
    # get mutated image
    src_image_transformed = ifft2(amplitude_src * np.exp(1j * phase_src))
    src_image_transformed = np.real(src_image_transformed)
    src_image_transformed=src_image_transformed.transpose(2,1,0)
    return src_image_transformed
# transform = A.Compose([A.FDA([trg_pth], p=1,beta_limit=0.005)])
def process(imgpth):
    image=np.array(Image.open(imgpth))
    # result = transform(image=image)['image']
    result = Image.fromarray(fda_butterworth(image))
    result.save(imgpth.replace('images', 'images_fda'))

if __name__=='__main__':
    # process('/data/fusai_release/train/images/0.tif')
    _ = mmcv.track_parallel_progress(process, img_paths, 10)
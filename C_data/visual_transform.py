import albumentations as A
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import mmcv
# transform = A.Compose([
#     # A.RandomBrightnessContrast(p=1.0),
#     # A.ShiftScaleRotate(shift_limit=0,
#     #     scale_limit=2,
#     #     rotate_limit=0,
#     #     p=1),
#     # A.OpticalDistortion()
#     # A.GaussianBlur(blur_limit=7, p=1.0)
#     # A.RGBShift(r_shift_limit=20,
#     #     g_shift_limit=20,
#     #     b_shift_limit=20,
#     #     p=1)
#     # A.GaussNoise(var_limit=(10.0, 40.0), p=1)
    
# ])
show_mask=1
img_dir="/data/chusai_release/train/images/"
ann_dir="/data/chusai_release/train/labels_9/"
img_id=180
image = cv2.imread(img_dir+str(img_id)+".tif")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
def n2gray(label):
    for i in range(len(label)):
        for j in range(len(label[0])):
            label[i][j]=int(float(label[i][j])*256/9)
    label=np.array(label,dtype=np.uint8)
    return label
if show_mask:
    mask = np.array(Image.open(ann_dir+str(img_id)+".png"))
    # transformed = transform(image=image, mask=mask)
    # img,tmask = transformed['image'],transformed['mask']
    size=512*3
    img=mmcv.imrescale(image,(size,size))#resize
    img=img[0:512,0:512,:]#crop 索引超界不报错
    img = mmcv.impad(img, shape=(512,512), pad_val=0)

    tmask=mmcv.imrescale(mask,(size,size), interpolation='nearest')
    tmask=tmask[0:512,0:512]
    tmask=mmcv.impad(tmask,shape=(512,512),pad_val=9)
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(image)
    plt.subplot(2,2,2)
    plt.imshow(img)
    plt.subplot(2,2,3)
    plt.imshow(n2gray(mask))
    plt.subplot(2,2,4)
    plt.imshow(n2gray(tmask))
else:
    # img = transform(image=image)['image']    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.imshow(img)
plt.savefig("visualize/c.png")
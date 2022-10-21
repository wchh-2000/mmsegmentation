from PIL import Image
import numpy as np
dir="/data/fusai_release/train/images/"
img=np.array(Image.open(dir+'4.tif'))
shiftn=50
lut=list(range(shiftn,256,1))+[255]*shiftn
lut=np.array(lut)
assert len(lut)==256
img=lut[img]
Image.fromarray(img.astype(np.uint8)).save("shift4.png")
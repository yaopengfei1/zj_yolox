import os
import cv2
import glob
import numpy as np


def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH))

root_dir= '../stitcher_img/'
sub_dir='20211119-181414/'
images = []
for i, file_name in enumerate(glob.glob(os.path.join(root_dir, sub_dir,"*.jpeg"))):
    img_path = f'{root_dir}/{sub_dir}/292_{str(i)}.jpeg'
    if img_path.__contains__('info'):
        continue
    image_data = cv2.imread(img_path)
    image_data=image_data[0: 2048, 320: 3072]
    image_data=rotate_bound(image_data,-2)
    image_data=cv2.resize(image_data,(image_data.shape[1]//2,image_data.shape[0]//2))
    print(img_path)
    # cv2.imshow(f"img {i}", image_data)
    # cv2.waitKey(0)
    images.append(image_data)
img_res=np.hstack(images)
cv2.imwrite('res3.jpeg', img_res)
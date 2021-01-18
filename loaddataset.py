import cv2
from pathlib import Path
import numpy as np

def load_image_folder(imgpath, imgext, res):

    path=Path(imgpath) # path to folder
    images=[]
    print(path)

    for imagepath in path.glob(imgext):
        img=cv2.imread(str(imagepath))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        images.append(img)
    return images
    #print(images)

def ligatures_dataset():
    data = np.load('./data/ligatures.npz')
    return (data['x_train'], data['x_test']), (data['y_train'], data['y_test'])


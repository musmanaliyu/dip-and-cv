
from numpy import asarray
from loaddataset import load_image_folder
import os
import numpy as np

class HWDataSet:
  images = None
  target = None # list cannot be initialized here!

def load_urduhw_dataset(path):
    hwdataset = HWDataSet()
    hwdataset.images = []
    hwdataset.target = []
    #for dirpath, dirs, files in os.walk(path):
    #    print(dirs)
    dirs = os.listdir(path)
    for dir in dirs:
        images = load_image_folder(path+'/'+dir, '*.png', 28)
        for image in images:
            hwdataset.target.append(dir)
            hwdataset.images.append(image)

    hwdataset.images = asarray(hwdataset.images)
    hwdataset.target = asarray(hwdataset.target)
    return hwdataset

# The digits dataset
xligatures = load_urduhw_dataset('./data/DataSet/Train')
yligatures = load_urduhw_dataset('./data/DataSet/Test')
np.savez_compressed('./data/ligatures', x_train=xligatures.images, x_test=xligatures.target, y_train=yligatures.images, y_test=yligatures.target)

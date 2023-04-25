import os
import zipfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import random
from PIL import Image
import shutil

workdir = Path(".")

# Define directory of training images
train_dir_images = workdir / 'data/HC18/training_set'

X_train_set = np.array([])
y_train_set = np.array([])

for image_name in train_dir_images.iterdir():
    if 'Annotation' in image_name.name:
        # Move to label folder
        dest_name = train_dir_images / 'label'
        shutil.move(image_name, dest_name / image_name.name)
        y_train_set = np.append(y_train_set, 'training_set/' + image_name.name)
    else:
        # Move to image folder
        dest_name = train_dir_images / 'image'
        shutil.move(image_name, dest_name / image_name.name)
        X_train_set = np.append(X_train_set, 'training_set/' + image_name.name)

X_train_set = np.sort(X_train_set) # list of training image strings
y_train_set = np.sort(y_train_set) # list of annotated training image strings
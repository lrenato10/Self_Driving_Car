#%%
import io
import os
from utils import get_dataset
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tensorflow as tf

#%%
path_data = "/home/Udacity/Self_Driving_Car/Object_Detection_Project/data/waymo/training_and_validation/*.tfrecord"
dataset = get_dataset(path_data)

#%% 
def display_images(batch):
    #plot image
    fig, ax=plt.subplots()
    img=batch["image"].numpy()
    ax.imshow(img)
    
    # mapping = {1: 'vehicle', 2: 'pedestrian', 4: 'cyclist'}
    color_mapping = {1: 'red', 2: 'blue', 4: 'green'}
    
    # plot bounding box with different colors
    for bb, label in zip (batch["groundtruth_boxes"].numpy(), batch["groundtruth_classes"].numpy()):
        #calculating bounding box
        y1, x1, y2, x2 = bb
        x1,x2=x1*img.shape[0], x2*img.shape[0]
        y1,y2=y1*img.shape[1], y2*img.shape[1]
        xy = (x1, y1)
        width = x2 - x1
        height = y2 - y1
        rec = patches.Rectangle(xy, width, height, linewidth=1, edgecolor=color_mapping[label], facecolor='none')
        ax.add_patch(rec)
    plt.axis('off')

    # Display 10 random images in dataset
plt.figure(figsize=(2, 2))
shuffle_data=dataset.shuffle(96, reshuffle_each_iteration=True).take(10)
for batch in shuffle_data:
        display_images(batch)
plt.show()

#%%
for i in range(10):
    a=i
    print('1')

# %%
# Display 10 random images in dataset
plt.figure(figsize=(2, 2))
shuffle_data=dataset.shuffle(96, reshuffle_each_iteration=True).take(10)
for batch in shuffle_data:
        display_images(batch)
plt.show()
# %%

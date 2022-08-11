# Object Detection in an Urban Environment

## Project overview
Object detection is a crucial  role in self-driving cars. In this project we will focus on two main parts, the dataset and the training and validation of our model.
The dataset has a key role in machine learning, because our model learns with supervised learning, so it will use our labeled data to generalize the information from the data. Due to this fact it is important to have representative images in many different conditions, which the car may find in real life and we also need to give a dataset with the number of objects per class as close as possible in order to avoid generating a bias in the model prediction.
Furthermore, the training and validation part is also a crucial part. Because there are several possibilities of different architectures and the engineer needs to choose the best one to improve the final performance playing with the trade-off of overfitting and underfitting. 

## Set up

### Requirements

To install the required packages there is a container file `Dockerfile` and a `requirements.txt` in the folder `build`, more details available in the README in this folder.


### Data

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/).

[OPTIONAL] - The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. We have already provided the data required to finish this project in the workspace, so you don't need to download it separately.
The first goal of this project is to download the data from the Waymo's Google Cloud bucket to your local machine. For this project, we only need a subset of the data provided (for example, we do not need to use the Lidar data). Therefore, we are going to download and trim immediately each file. In `download_process.py`, you can view the `create_tf_example` function, which will perform this processing. This function takes the components of a Waymo Tf record and saves them in the Tf Object Detection api format. An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records). We are already providing the `label_map.pbtxt` file.

You can run the script using the following command:
```
python download_process.py --data_dir {processed_file_location} --size {number of files you want to download}
```

### Structure

The data used for training, validation and testing is organized in `data` folder as follow:

```
data/waymo/
    - training_and_validation - contains 96 files to train and validate your models
    - train: contain the train data
    - val: contain the val data
    - test - contains 3 files to test your model and create inference videos
```
The `training_and_validation` folder contains file that have been downsampled: we have selected one every 10 frames from 10 fps videos. The `testing` folder contains frames from the 10 fps video without downsampling.
```

To split this `training_and_validation` data into `train`, and `val` you can use the `create_splits.py` file.

The `experiments` folder is organized as follow:
```
experiments/
    - label_map.pbtxt
    - reference/... reference training with the unchanged config file
    - exporter_main_v2.py: to create an inference model
    - model_main_tf2.py: to launch training
    - experiment0/... create a new folder for each experiment runned
    - experiment1/...
    - experimentX/...
    - pretrained-model/: contains the checkpoints of the pretrained models (FRCNN  and SDD).
```


##Dataset

### Dataset Analysis

In the `Exploratory Data Analysis` notebook it is possible to explore the dataset from the tfrecord format. By using the `display_images` function it is possible to visualize a frame of the dataset with the bounding box with different colors for each class, reto to vehicle, blue to pedestrian and green to cyclist. Hence, it is possible to use this function to visualize the imagens from the dataset and analyze the quality of them.

We can see the result of this function displaying 10 images in the figures below:

![img1](images/img1.png)![img2](images/img2.png)![img3](images/img3.png)![img4](images/img4.png)![img5](images/img5.png)![img6](images/img6.png)![img7](images/img7.png)![img8](images/img8.png)![img9](images/img9.png)![img10](images/img10.png)


After see many different images from this dataset we can conclude that all images are visible, with day, night, sunny and rainy conditions. Furthermore, without miss classification and repeated bounding boxes for the same object. We also visualized the three classes with their respective colors, however it is visible the higher quantity of vehicles.

In the `Exploratory Data Analysis` notebook there is also a statistical analysis of the data. In the following plots we can see the quantity of objects for each class, the distribution of vehicles, pedestrians and cyclists, and there is also plots with the mean bounding box width, height and area for each class.
With this data we can conclude that cars outnumber pedestrians, and pedestrians outnumber cyclists. This information is very relevant because it can bias the model to classify the object as a vehicle because it is more probable. We can also conclude that there are normally between 0 and 35 vehicles per image. And for cyclists and pedestrians normally there aren't any.

![img11](images/img11.png)![img12](images/img12.png)![img13](images/img13.png)![img14](images/img14.png)![img15](images/img15.png)![img16](images/img16.png)


### Create the training - validation splits
To split the dataset in three folder for cross-validation there is the `create_splits.py` file. This code will take all files in the `destination folder` and will create a  `destination folder `to store a folder for train data, another for validation and another for test.
The proportion of validation and test can be passed as parameters when executing this function. This split function will randomly distribute all the data from the `source folder` to the `destination folder` respecting the selected proportion.
 
```
python create_splits.py --destination data/waymo/splits --source data/waymo/training_and_validation_and_test --pval proportion_of_validation --ptest proportion_of_test
```


### Config file of the model

To train this model we will use a `pipeline.config` file, this file contains all the information to train and validate the model. We have the `pipeline.config` that has a SSD object detection architecture and `pipelinefrcnn.config` that has a Faster-RCNN object detection architecture.
Due to the complexity of the model and the similarity with other models we will use the fine tune to start our model, because it is not worth starting from scratch, there are many parameters to find and we can use the previous knowledge of another model to start our model.

To import the SSD pretrained model we can execute the following commands
```
cd experiments/pretrained_model/
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
tar -xvzf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
rm -rf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
```

To import the Faster-RCNN pretrained  model we can execute the following commands
```
cd experiments/pretrained_model/
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz
tar -xvzf faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz
rm -rf faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz
```

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`. Copy `pipeline_new.config` in experiments/reference/.
We will use this pipeline to train our model.

### Training

You will now launch your very first experiment with the Tensorflow object detection API. Move the `pipeline_new.config` to the `/home/workspace/experiments/reference` folder. Now launch the training process:
* a training process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
Once the training is finished, launch the evaluation process:
* an evaluation process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```

**Note**: Both processes will display some Tensorflow warnings, which can be ignored. You may have to kill the evaluation script manually using
`CTRL+C`.

To monitor the training, you can launch a tensorboard instance by running `python -m tensorboard.main --logdir experiments/reference/`. You will report your findings in the writeup.

### Improve the performances

Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup.

Keep in mind that the following are also available:
* experiment with the optimizer: type of optimizer, learning rate, scheduler etc
* experiment with the architecture. The Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) offers many architectures. Keep in mind that the `pipeline.config` file is unique for each architecture and you will have to edit it.

**Important:** If you are working on the workspace, your storage is limited. You may to delete the checkpoints files after each experiment. You should however keep the `tf.events` files located in the `train` and `eval` folder of your experiments. You can also keep the `saved_model` folder to create your videos.


### Creating an animation
#### Export the trained model
Modify the arguments of the following function to adjust it to your models:

```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/
```

This should create a new folder `experiments/reference/exported/saved_model`. You can read more about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model).

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path /data/waymo/testing/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif
```

## Submission Template

### Project overview
This section should contain a brief description of the project and what we are trying to achieve. Why is object detection such an important component of self driving car systems?

### Set up
This section should contain a brief description of the steps to follow to run the code for this repository.

### Dataset
#### Dataset analysis
This section should contain a quantitative and qualitative description of the dataset. It should include images, charts and other visualizations.
#### Cross validation
This section should detail the cross validation strategy and justify your approach.

### Training
#### Reference experiment
This section should detail the results of the reference experiment. It should includes training metrics and a detailed explanation of the algorithm's performances.

#### Improve on the reference
This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.

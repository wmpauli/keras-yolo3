# keras-yolo3 (AzureML)

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

**Note:** This fork of the keras-yolo3 was create with the sole intent of demonstrating how to train a keras-yolo3 model on the VOC Pascal dataset, using [AzureML](https://azure.microsoft.com/en-us/services/machine-learning/).  If you are familiar with the original repository, you may want to jump right to section `Train On AzureML` below.

## Introduction

A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).


---

## Quick Start

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Run YOLO detection.

```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
python yolo_video.py [OPTIONS...] --image, for image detection mode, OR
python yolo_video.py [video_path] [output_path (optional)]
```

For Tiny YOLOv3, just do in a similar way, just specify model path and anchor path with `--model model_file` and `--anchors anchor_file`.

### Usage
Use --help to see usage of yolo_video.py:
```
usage: yolo_video.py [-h] [--model MODEL] [--anchors ANCHORS]
                     [--classes CLASSES] [--gpu_num GPU_NUM] [--image]
                     [--input] [--output]

positional arguments:
  --input        Video input path
  --output       Video output path

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file, default model_data/yolo.h5
  --anchors ANCHORS  path to anchor definitions, default
                     model_data/yolo_anchors.txt
  --classes CLASSES  path to class definitions, default
                     model_data/coco_classes.txt
  --gpu_num GPU_NUM  Number of GPU to use, default 1
  --image            Image detection mode, will ignore all positional arguments
```
---

4. MultiGPU usage: use `--gpu_num N` to use N GPUs. It is passed to the [Keras multi_gpu_model()](https://keras.io/utils/#multi_gpu_model).

## Training

1. Generate your own annotation file and class names file.  
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    For VOC dataset, try `python voc_annotation.py`  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

2. Make sure you have run `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`  
    The file model_data/yolo_weights.h5 is used to load pretrained weights.

3. Modify train.py and start training.  
    `python train.py`  
    Use your trained weights or checkpoint weights with command line option `--model model_file` when using yolo_video.py
    Remember to modify class path or anchor path, with `--classes class_file` and `--anchors anchor_file`.

If you want to use original pretrained weights for YOLOv3:  
    1. `wget https://pjreddie.com/media/files/darknet53.conv.74`  
    2. rename it as darknet53.weights  
    3. `python convert.py -w darknet53.cfg darknet53.weights model_data/darknet53_weights.h5`  
    4. use model_data/darknet53_weights.h5 in train.py

## Train on AzureML

### Prerequisites

**Create Conda Environemtn**

For you convenience, we added a Conda Environment definition. We recommend you [install Miniconda](https://docs.conda.io/en/latest/miniconda.html) for your distribution, to then create a conda environment for this project:

```
conda env create -f environment.yml
```

**Configure your AzureML Workspace**

Create a file `aml/config.json` with the information regarding your Azure subscription and your AML Workspace. The file should look like this:

```
{
    "workspace_name": <>,
    "subscription_id": <>,
    "resource_group": <>,
    "location": <>
}
```

**Download pretrained Yolo v3 weights**

Please download the weights for the pretrained tiny yolov3 model, and store them in the root of this repository (filename should be `yolov3-tiny.weights`).

> Please also use follow the above instructions to convert the weights to the format expected by this repo (`convert.py`).

**Download VOC Pascal dataset (Dev Kit)**

The dataset is available here: [http://host.robots.ox.ac.uk/pascal/VOC/voc2007/#devkit](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/#devkit)

Direct download links:
- [http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
- (optional) [http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)

Please unpack these tar balls into the root directory of this repository (i.e. `/<path>/<to>/<repo>/VOCdevkit/`).

> Please also use follow the above instructions to convert the dataset to the format expected by this repo (`voc_annotation.py`).

### Create AML workspace and upload data to Azure Blob Storage

This can be accomplished by executing the following:

```
conda activate keras-yolo3
python aml/upload_data.py
```

### Submit Run for Training on AML Compute

```
conda activate keras-yolo3
python aml/train_aml.py
```

## Change log

We made the following changes to enable training on AzureML.
- Small changes to `voc_annotation.py` and `train.py` to enable loading data from Azure Blob Storage
- We added a callback for logging results to the AML workspace. This was done in `train.py`:

```
from keras.callbacks import Callback

class LogRunMetrics(Callback):
    # callback at the end of every epoch
    def on_epoch_end(self, epoch, log):
        # log a value repeated which creates a list
        run.log("val_loss", log["val_loss"])
        run.log("loss", log["loss"])
```
- We added code for model registration, after successful training (in `train.py`):
```
run.register_model("tiny_yolov3", model_path=log_dir + 'trained_weights_final.h5')
```
- We did some other minor changes. For a complete list, we recommend using GitHubs compare function to compare our repository with the original.

---

## Some issues to know

1. The test environment is
    - Python 3.5.2
    - Keras 2.1.5
    - tensorflow 1.6.0

2. Default anchors are used. If you use your own anchors, probably some changes are needed.

3. The inference result is not totally the same as Darknet but the difference is small.

4. The speed is slower than Darknet. Replacing PIL with opencv may help a little.

5. Always load pretrained weights and freeze layers in the first stage of training. Or try Darknet training. It's OK if there is a mismatch warning.

6. The training strategy is for reference only. Adjust it according to your dataset and your goal. And add further strategy if needed.

7. For speeding up the training process with frozen layers train_bottleneck.py can be used. It will compute the bottleneck features of the frozen model first and then only trains the last layers. This makes training on CPU possible in a reasonable time. See [this](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for more information on bottleneck features.

## Contribute

We invite anybody to provide constructive feedback via github issues.  We also welcome pull requests.

## Disclaimer

This repository is not officially maintained.
# "Oh Hell" Scorekeeper

## Promising articles

https://towardsdatascience.com/training-yolo-for-object-detection-in-pytorch-with-your-custom-dataset-the-simple-way-1aa6f56cf7d9
https://towardsdatascience.com/object-detection-and-tracking-in-pytorch-b3cf1a696a98
https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data

## Setup

```bash
curl https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz --output dtd.tar.gz
tar xfvz dtd.tar.gz
python3 -m pip install \
    numpy opencv-python Shapely \
    torch torchvision
```

## Dataset generation

https://commons.wikimedia.org/wiki/Category:Complete_decks_of_playing_cards_laid_out#/media/File:Color_52_Faces_v.2.0.svg

sourced from

https://commons.wikimedia.org/wiki/Category:Complete_decks_of_playing_cards_laid_out

convert cards

```bash
qlmanage -t -s 2048 -o . Color_52_Faces_v.2.0.svg
```

## Environment setup

```bash
python -m venv /path/to/new/virtual/environment
```

## train.py usage

    usage: train.py [-h] [--epochs EPOCHS] [--image_folder IMAGE_FOLDER] [--batch_size BATCH_SIZE]
                    [--model_config_path MODEL_CONFIG_PATH] [--data_config_path DATA_CONFIG_PATH]
                    [--weights_path WEIGHTS_PATH] [--class_path CLASS_PATH] [--conf_thres CONF_THRES]
                    [--nms_thres NMS_THRES] [--n_cpu N_CPU] [--img_size IMG_SIZE]
                    [--checkpoint_interval CHECKPOINT_INTERVAL] [--checkpoint_dir CHECKPOINT_DIR] [--use_cuda USE_CUDA]

    optional arguments:
    -h, --help            show this help message and exit
    --epochs EPOCHS       number of epochs
    --image_folder IMAGE_FOLDER
                            path to dataset
    --batch_size BATCH_SIZE
                            size of each image batch
    --model_config_path MODEL_CONFIG_PATH
                            path to model config file
    --data_config_path DATA_CONFIG_PATH
                            path to data config file
    --weights_path WEIGHTS_PATH
                            path to weights file
    --class_path CLASS_PATH
                            path to class label file
    --conf_thres CONF_THRES
                            object confidence threshold
    --nms_thres NMS_THRES
                            iou thresshold for non-maximum suppression
    --n_cpu N_CPU         number of cpu threads to use during batch generation
    --img_size IMG_SIZE   size of each image dimension
    --checkpoint_interval CHECKPOINT_INTERVAL
                            interval between saving model weights
    --checkpoint_dir CHECKPOINT_DIR
                            directory where model checkpoints are saved
    --use_cuda USE_CUDA   whether to use cuda if available

## train.py example

```bash
DS_ROOT=${HOME}/projects/oh-hell-scorekeeper/data/pytorch/v0
WEIGHTS_PATH=${HOME}/projects/yolov3.weights
curl https://pjreddie.com/media/files/yolov3.weights --output ${WEIGHTS_PATH}
python3 train.py \
    --batch_size 4 \
    --model_config_path ${DS_ROOT}/../../yolov3.cfg \
    --data_config_path ${DS_ROOT}/config/coco.data \
    --class_path ${DS_ROOT}/config/coco.names \
    --weights_path ${WEIGHTS_PATH} \
    --image_folder ${DS_ROOT}/images/ \
    --checkpoint_dir ${DS_ROOT}/checkpoints/ \
    --use_cuda false
```

# "Oh Hell" Scorekeeper

## Promising articles

https://towardsdatascience.com/training-yolo-for-object-detection-in-pytorch-with-your-custom-dataset-the-simple-way-1aa6f56cf7d9
https://towardsdatascience.com/object-detection-and-tracking-in-pytorch-b3cf1a696a98
https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data

## yolov5 stuff

https://github.com/ultralytics/yolov5
https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

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

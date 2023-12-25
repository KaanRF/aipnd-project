# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code
for an image classifier built with PyTorch, then convert it into a command line application.

# Information

- If you want to use 'gpu' option then be sure you have installed the pytorch for CUDA compilation
- If you are using anaconda, then install pip to use given requirements.txt
- 'console_app' is using project directory to save checkpoints
- 'console_app' will try to avoid retrain same arch, please follow to instruction

# Commands

## Train

### Help

``` python train.py -h
python train.py -h

usage: train.py [-h] [--data_dir DATA_DIR] [--save_dir SAVE_DIR] [--arch {vgg19,alexnet,resnet}] [--learning_rate LEARNING_RATE] [--hidden_units HIDDEN_UNITS] [--epochs EPOCHS] [--gpu]

Train a new network on a dataset

options:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Path to the directory containing the data
  --save_dir SAVE_DIR   Directory to save checkpoints
  --arch {vgg19,alexnet,resnet}
                        Model architecture
  --learning_rate LEARNING_RATE
                        Learning rate
  --hidden_units HIDDEN_UNITS
                        Number of hidden units
  --epochs EPOCHS       Number of epochs
  --gpu                 Use GPU for training
```

### Basic usage

- `python train.py --data_dir ../flowers/` '../flowers/' is up to your setup !

### Customized usage

- `python train.py --data_dir ../flowers/ --arch "alexnet" --learning_rate 0.02 --hidden_units 512 --epochs 15 --gpu`

## Predict

### Help

```
`python predict.py -h`

usage: predict.py [-h] [--image_path IMAGE_PATH] [--checkpoint CHECKPOINT] [--top_k TOP_K] [--category_names CATEGORY_NAMES] [--gpu]

Predict flower name from an image

options:
  -h, --help            show this help message and exit
  --image_path IMAGE_PATH
                        path to the input image
  --checkpoint CHECKPOINT
                        path to the checkpoint file
  --top_k TOP_K         return top K most likely classes
  --category_names CATEGORY_NAMES
                        path to the category names mapping file
  --gpu                 use GPU for inference
```

### Basic usage

- `python predict.py ../flowers/train/11/image_03095.jpg checkpoint` '../flowers/train/11/image_03095.jpg' is up to your
  setup !

### Customized usage

- `python predict.py input checkpoint --category_names cat_to_name.json`
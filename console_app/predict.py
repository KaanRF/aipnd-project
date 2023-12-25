import argparse
import json
import sys

import torch

from image_preprocessor import ImagePreprocessor
from model_helper import ModelHelper
from predict_image import PredictImage
from utility import is_any_checkpoint_file_exist

parser = argparse.ArgumentParser(description='Predict flower name from an image')
parser.add_argument('--image_path', help='path to the input image')
parser.add_argument('--checkpoint', help='path to the checkpoint file')
parser.add_argument('--top_k', type=int, default=5, help='return top K most likely classes')
parser.add_argument('--category_names', default='cat_to_name.json', help='path to the category names mapping file')
parser.add_argument('--gpu', action='store_true', help='use GPU for inference')

args = parser.parse_args()

if args.image_path is None:
    print("Please select 'image_path', stopping application...")
    sys.exit()

if args.checkpoint is None:
    print("Checkpoints has not been selected, using default path")
    args.checkpoint = 'checkpoints'

if is_any_checkpoint_file_exist(args.checkpoint) is None:
    print(f"There is no any trained model found under {args.checkpoint}, See the 'python train.py -h'")

check_point_file = is_any_checkpoint_file_exist(args.checkpoint)

if check_point_file is None:
    print("No any checkpoint file is found, please first train...")

print(f"Using checkpoint file {check_point_file}")

loaded_checkpoint = torch.load(check_point_file)

# Build the model with check point
model_helper = ModelHelper(loaded_checkpoint['architecture'], loaded_checkpoint['hidden_units'])
model_helper.get_model().load_state_dict(loaded_checkpoint['model_state_dict'])
model_helper.get_model().class_to_idx = loaded_checkpoint['class_to_idx']

print(f"Initialized model:\n {model_helper.get_model()}")

preprocessed_image = ImagePreprocessor(args.image_path)

predict_image = PredictImage(preprocessed_image.get_preproccessed_image(), model_helper.get_model(), args.top_k,
                             args.gpu)
probabilities, classes = predict_image.get_prediction_outputs()
print("Probabilities {}".format(probabilities))
print("Classes {}".format(classes))

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

predict_image.display_predicted_image(args.image_path, cat_to_name)

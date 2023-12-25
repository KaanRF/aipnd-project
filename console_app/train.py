import argparse
from dataset_loader import DataLoader
from model_helper import ModelHelper
from utility import check_pth_file
import sys
import keyboard

# Define command line arguments
parser = argparse.ArgumentParser(description='Train a new network on a dataset')
parser.add_argument('--data_dir', type=str, help='Path to the directory containing the data')
parser.add_argument('--save_dir', type=str, help='Directory to save checkpoints')
parser.add_argument('--arch', type=str, default='vgg19', choices=['vgg19', 'alexnet', 'resnet'],
                    help='Model architecture')
parser.add_argument('--learning_rate', type=float, default=0.003, help='Learning rate')
parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

args = parser.parse_args()

if args.data_dir is None:
    print("Please select 'data_dir', Stopping application...")
    sys.exit()

if args.save_dir is None:
    print("'save_dir' is not selected, using default path 'checkpoints' .")
    args.save_dir = 'checkpoints'

print(args.data_dir, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)

# Ask user if already trained model existed
if check_pth_file(args.save_dir, args.arch) is not None:
    print(
        f"There is already trained model for the selected arch {args.arch}, If you want you want you can start "
        f"prediction")
    answer = input("Press 'c' to continue train, Press 'q' to stop \n")
    if answer == 'q':
        print("See the command for start prediction 'python predict.py -h'")
        sys.exit()
    if answer == 'c':
        print("Continue with new training")


# Load the dataset with given path
dataset_loader = DataLoader(args.data_dir)

train_dataset, test_dataset, valid_dataset = dataset_loader.get_datasets()
train_loader, test_loader, valid_loader = dataset_loader.get_loaders()

# Create the model and train
model_helper = ModelHelper(args.arch, args.hidden_units)

print(
    f"Starting to train model with given arguments gpu = {args.gpu}, epoch number = {args.epochs}, "
    f"learn rate = {args.learning_rate}")
model_helper.train_model(train_loader, valid_loader, args.gpu, args.epochs, args.learning_rate)

print(
    f"Training finished, now testing the model with given arguments gpu = {args.gpu}")
model_helper.test_model(test_loader, args.gpu)

print("Saving the model checkpoints")
model_helper.save_checkpoint(args.save_dir, train_dataset, args.epochs, args.learning_rate)

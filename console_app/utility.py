import os
import sys

import keyboard

def check_pth_file(directory, arch):
    # Get the list of files in the directory
    current_directory = os.getcwd()
    files = os.listdir(current_directory + '/' + directory)

    # Check each file for the .pth extension
    for file in files:
        # Skip files that start with '.'
        if file.startswith('.'):
            continue

        if file.endswith(".pth"):
            # Construct the full path of the file
            file_path = os.path.join(directory, file)
            if file.find(arch) != -1:
                print("Found checkpoint file for the arch {}".format(arch))
                return file_path

    # If no .pth file is found, return None
    return None





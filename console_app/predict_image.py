import torch
from torch import nn
from torch import optim
from torchvision import models
import logging
import matplotlib.pyplot as plt
import time
import numpy as np


class PredictImage:
    def __init__(self, preprocessed_image, model, topk, is_gpu):
        super().__init__()

        self.topk = topk
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_gpu and device == "cpu":
            logging.warning("GPU is requested to be used, but it is not available on your system.")

        logging.warning("Using device is {}".format(device))

        self._preprocessed_image = preprocessed_image.unsqueeze(0)
        self._preprocessed_image = self._preprocessed_image.to(device)

        model.eval()

        # Forward pass
        with torch.no_grad():
            output = model(input)
            probabilities = torch.exp(output)

        # Get the top k probabilities and classes
        self.topk_probs, self.topk_indices = torch.topk(probabilities, k=self.topk)

        # Convert the indices to class labels
        idx_to_class = {idx: class_ for class_, idx in model.class_to_idx.items()}
        self.topk_classes = [idx_to_class[idx.item()] for idx in self.topk_indices.squeeze()]
        self.topk_probs = self.topk_probs.squeeze().tolist()

    def _imshow(image, ax=None, title=None):
        """Imshow for Tensor."""
        if ax is None:
            fig, ax = plt.subplots()

        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
        image = image.numpy().transpose((1, 2, 0))

        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean

        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)

        ax.imshow(image)

        return ax

    def get_prediction_outputs(self):
        return self.topk_probs, self.topk_classes

    def display_predicted_image(self, cat_to_name):
        class_labels = [cat_to_name[class_] for class_ in self.topk_classes]
        # Plot the image and the bar graph
        fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=1, nrows=2)

        # Plot the image
        ax1.axis('off')
        self._imshow(self._preprocessed_image, ax1, title="Image")

        # Plot the bar graph
        ax2.barh(np.arange(self.topk), self.topk_probs)
        ax2.set_aspect(0.1)
        ax2.set_yticks(np.arange(self.topk))
        ax2.set_yticklabels(class_labels)
        ax2.set_xlim(0, 1.1)
        ax2.invert_yaxis()
        ax2.set_xlabel('Probability')

        plt.tight_layout()
        plt.show()



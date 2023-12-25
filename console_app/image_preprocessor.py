from PIL import Image
from torchvision import transforms


class ImagePreprocessor:
    def __init__(self, image_path):
        super().__init__()

        image = Image.open(image_path)

        # Define the image transformation
        image_transformation = transforms.Compose([transforms.Resize(255),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225])])

        self._preprocessed_image = image_transformation(image)

    def get_preproccessed_image(self):
        return self._preprocessed_image

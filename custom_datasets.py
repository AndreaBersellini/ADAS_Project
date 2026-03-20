from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image

class SirenDetectionDataset(Dataset):
    def __init__(self, spectrograms : np.ndarray, classes : np.ndarray):
        self._images = spectrograms
        self._labels = classes

    def transform(self, image : np.ndarray, label: int):
        # 1 CHANNEL CONVERSION
        image = Image.fromarray(image).convert('L')

        # RESIZE
        resize = transforms.Resize((128, 128))
        image = resize(image)

        # CONVERT TO TENSOR AND NORMALIZE (0,1)
        image = TF.to_tensor(image)

        return image, label

    def __getitem__(self, index : int):
        image = self._images[index]
        label = self._labels[index]
        x, y = self.transform(image, label)
        return x, y

    def __len__(self):
        return len(self._images)
    
class SirenRegressionDataset(Dataset):
    def __init__(self, spectrograms : np.ndarray, velocities : np.ndarray):
        self._images = spectrograms
        self._velocities = velocities

    def transform(self, image : np.ndarray, label: int):
        # 1 CHANNEL CONVERSION
        image = Image.fromarray(image).convert('L')

        # RESIZE
        resize = transforms.Resize((128, 128))
        image = resize(image)

        # CONVERT TO TENSOR AND NORMALIZE (0,1)
        image = TF.to_tensor(image)
    
        return image, label
    
    def __getitem__(self, index : int):
        image = self._images[index]
        label = self._velocities[index]
        x, y = self.transform(image, label)
        return x, y
    
    def __len__(self):
        return len(self._images)
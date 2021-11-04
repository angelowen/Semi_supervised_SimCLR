from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator

from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np

np.random.seed(0)

class PlantSeedlingDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.x = []
        self.y = []
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.transform = transforms.Compose([ transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * 224)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        if self.root_dir.name == 'unlabel':
            for file in self.root_dir.glob('*'):
                self.x.append(file)
                
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open(self.x[index]).convert('RGB')
        img1 = self.transform(image) 
        img2 = self.transform(image)
        return [img1,img2] # return 2 data augmentation

class PlantFinetuneDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.x = []
        self.y = []
        self.transform = transform
        self.num_classes = 0

        if self.root_dir.name == 'train':
            for i, _dir in enumerate(self.root_dir.glob('*')):
                for file in _dir.glob('*'):
                    self.x.append(file)
                    self.y.append(i)

                self.num_classes += 1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open(self.x[index]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.y[index]


import os
import csv
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets

def animalFaceDataset(
        split: str='train',
        transform: transforms=None,
        imgSize: int=128
    ):
        root = './afhq' if 'afhq' in os.listdir(os.getcwd()) else './dataset/afhq'
        splitPath = os.path.join(root, split)
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(imgSize),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5)
                )
            ])
        dataset = datasets.ImageFolder(
            root=splitPath,
            transform=transform
        )
        
        return dataset

class animalFaceDatasetCSV(Dataset):
    def __init__(
        self,
        split: str='train',
        transform: transforms=None,
        imgSize: int=128
    ):
        self.root = './afhq' if 'afhq' in os.listdir(os.getcwd()) else './dataset/afhq'
        self.transform = None
        if transform is None:
            self.transfrom = transforms.Compose([
                transforms.Resize(imgSize),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5)
                )
            ])
        self.metadata = pd.read_csv(os.path.join(self.root, 'afhq-metadata.csv'))
        self.metadata = self.metadata[split in self.metadata['split']]
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata.iloc[idx]
        image = Image.open(os.path.join(self.root, item['filePath']))
        image = self.transfrom(image)
        itemDict = dict(className=item['class'], classIdx=['classIdx'], image=image)
        
        return itemDict

def writeMetadataCSV() -> None:
    splits = ['train', 'val']
    dataClass = ['cat', 'dog', 'wild']
    root = './afhq' if 'afhq' in os.listdir(os.getcwd()) else './dataset/afhq'
    
    if 'afhq-metadata.csv' in os.listdir(root):
        print("File already exists!")
        return
    
    meta = open(os.path.join(root, 'afhq-metadata.csv'), 'w', newline='')
    metaWriter = csv.writer(meta)
    metaWriter.writerow(['split', 'class', 'classIdx', 'filePath'])
    
    for split in splits:
        splitPath = os.path.join(root, split)
        for idx, animal in enumerate(dataClass):
            animalPath = os.path.join(splitPath, animal)
            for dataPath in os.listdir(animalPath):
                newData = [split, animal, idx, os.path.join(split, animal, dataPath)]
                metaWriter.writerow(newData)
    return            
    
if __name__ == "__main__":
    writeMetadataCSV()
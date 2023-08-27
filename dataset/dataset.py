import os
import csv

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
            for dataPath in os.listdir(os.path.join(splitPath, animal)):
                newData = [split, animal, idx, dataPath]
                metaWriter.writerow(newData)
    return            
    
if __name__ == "__main__":
    writeMetadataCSV()
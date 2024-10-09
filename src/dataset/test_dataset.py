import numpy as np
from os.path import join, exists
from os import listdir

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from PIL import Image
import h5py


def read_coordinates(image_names):    # -> shape = (len(image_names), 2])
    all_coordinates = []
    for i in image_names:
        coor = read_coordinate(i)
        all_coordinates.append(coor)
    return np.array(all_coordinates)
    
def read_coordinate(image_name):
    parts = image_name.split("@")
    x = parts[1]
    y = parts[2]
    return [float(x), float(y)]


def collate_fn(batch):
    """
    Custom batch preparation function. Must be used with TestDataset in DataLoaders.
    Collect images_high, images_low, and locations along batch dimension and stack the former two into tensors if possible.
    """
    images_high = [batch_item[0] for batch_item in batch]
    images_low = [batch_item[1] for batch_item in batch]
    locations = torch.stack([batch_item[2] for batch_item in batch])
    if all(isinstance(image_high, torch.Tensor) for image_high in images_high) and \
        len(set(tuple(image_high.shape) for image_high in images_high)) == 1:    # if all tensors have the same shape
        images_high, images_low = torch.stack(images_high), torch.stack(images_low)
    return images_high, images_low, locations


class TestDataset(Dataset):

    def __init__(self, image_folder, resolution, train_loss_config, neighbors_suffix=None):
        """
        - image_folder: path to query images folder (one level above individual resolutions folder)
        - resolution: specifies the resolution folder to choose for query images
        - neighbors_suffix: path of neighbors file relative to image_folder
        """
        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
        ])
        self.image_folder = image_folder
        self.resolution = resolution 
        self.nPosSample,self.nNegSample = train_loss_config['nPosSample'],train_loss_config['nNegSample'] # 1, 5
        self.__prepare_data(image_folder, neighbors_suffix)


    def __getitem__(self, index):
        """
        Retrieves a high and a low resolution image, each with its postive and negative neighboring images at its resolution.
        All image are prepended with a dimension of length 1.
        """
        high_image_set, low_image_set=self.data[index]

        images_high, images_low, locations=[], [], []
        image_high_path, positives_high, negtives_high = high_image_set
        image_low_path, positives_low, negtives_low = low_image_set
        # Only stack image tensors if all have the same shape; otherwise keep as list of tensors
        img_same_res = True
        for imp in [image_high_path] + positives_high + negtives_high:
            im = Image.open(imp)
            if im.mode != 'RGB':
                im = im.convert('RGB')
            im = self.input_transform(im).unsqueeze(0)
            if len(images_high) > 0 and im.shape != images_high[-1].shape: img_same_res = False
            images_high.append(im)
        if img_same_res: images_high = torch.stack(images_high)
        for imp in [image_low_path] + positives_low + negtives_low:
            im = Image.open(imp)
            if im.mode != 'RGB':
                im = im.convert('RGB')
            im = self.input_transform(im).unsqueeze(0)
            images_low.append(im)
            locations.append(read_coordinate(imp))
        if img_same_res: images_low = torch.stack(images_low)

        locations = np.array(locations)
        locations = torch.tensor(locations)

        return [images_high, images_low, locations]
        # Shape: [(7, 1, chnl, height, with), (7, 1, ..., ..., ...), (7, 2)]

    def __len__(self):
        return len(self.data)

    def __prepare_data(self, image_folder, neighbors_suffix):
        self.data, neighbor_file = [], None if neighbors_suffix is None else h5py.File(join(image_folder, neighbors_suffix), "r")

        raw_folder = join(image_folder, 'raw')
        image_list=listdir(raw_folder)
        self.image_coordinates = read_coordinates(image_list)

        for image in image_list:
            if neighbor_file is None or image in neighbor_file:
                image_high_path=join(image_folder, 'raw', image)
                image_low_path=join(image_folder, self.resolution, image)

                if neighbor_file is None:
                    self.data.append([[image_high_path, [], []], [image_low_path, [], []]])
                    continue
                
                positives_high,negatives_high=[],[]
                positives_low,negatives_low=[],[]
                
                positives_pool=neighbor_file[image]['positives'][:] #(20,)
                negatives_pool=neighbor_file[image]['negtives'][:] #(100,)

                ind=0
                while len(positives_high) < self.nPosSample:
                    name=positives_pool[ind].decode('utf-8')
                    positive_high_path=join(image_folder, 'raw', name)
                    positive_low_path=join(image_folder, self.resolution, name)
                    if exists(positive_high_path) and exists(positive_low_path):
                        positives_high.append(positive_high_path)
                        positives_low.append(positive_low_path)
                        
                    ind += 1
                    if ind==len(positives_pool)-1:
                        break

                ind=0
                while len(negatives_high) < self.nNegSample:
                    name=negatives_pool[ind].decode('utf-8')
                    negative_high_path=join(image_folder, 'raw', name)
                    negative_low_path=join(image_folder, self.resolution, name)
                    if exists(negative_high_path) and exists(negative_low_path):
                        negatives_high.append(negative_high_path)
                        negatives_low.append(negative_low_path)
                        
                    ind += 1
                    if ind==len(negatives_pool)-1:
                        break
                if len(positives_high)==self.nPosSample and len(negatives_high)==self.nNegSample:
                    self.data.append([[image_high_path,positives_high,negatives_high],[image_low_path,positives_low,negatives_low]])
        if neighbor_file is not None:
            neighbor_file.close()

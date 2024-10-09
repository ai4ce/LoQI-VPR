from os.path import join, exists, basename, isdir
from os import listdir
import multiprocessing
import numpy as np
import json
from torch.utils.data import Dataset,Sampler
import json
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import h5py
import random

from dataset.gsv_cities_data import convert_longlat_to_utm

def chunk(indices, size):
    return torch.split(torch.tensor(indices),size)

class BatchSampler(Sampler):
    def __init__(self, data_indices, batch_size):
        self.data_indices=data_indices
        self.batch_size=batch_size
    
    def __iter__(self):
        data_batches=[]
        for d in self.data_indices:
            random.shuffle(d)
            data_batches.append(chunk(d,self.batch_size))
        d=data_batches[0]
        for dd in data_batches[1:]:
            d+=dd
        all_batches=list(d)
        all_batches=[batch.tolist() for batch in all_batches]
        random.shuffle(all_batches)
        return iter(all_batches)
    
    def __len__(self):
        return sum([len(d) for d in self.data_indices])/self.batch_size
    
class UNav_dataset(Dataset):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    def __init__(self,image_folder,resolution,qp,pitch_list,yaw_list,**config):
        self.nPosSample,self.nNegSample=config['nPosSample'],config['nNegSample']
        num=0
        for pitch in pitch_list:
            for yaw in yaw_list:
                neighbor_folder=join(image_folder,pitch,yaw,'neighbor',resolution,str(qp))
                num+=len(listdir(neighbor_folder))
        self.data=np.empty(num,dtype='U100')
        index=0
        for pitch in pitch_list:
            for yaw in yaw_list:
                neighbor_folder=join(image_folder,pitch,yaw,'neighbor',resolution,str(qp))
                datas=listdir(neighbor_folder)
                for data in datas:
                    self.data[index]=join(neighbor_folder,data)
                    index+=1
        
    def __getitem__(self, index):
        with open(self.data[index],'r') as f:
            data=json.load(f)
        locs=data['locs']
        data=data['paths']
        images_high_dict=data['high']
        image_high_path=images_high_dict['path']
        positives_high=images_high_dict['positives']
        negtives_high=images_high_dict['negtives']

        images_low_dict=data['low']
        image_low_path=images_low_dict['path']
        positives_low=images_low_dict['positives']
        negtives_low=images_low_dict['negtives']

        images_low_path,images_high,images_low=[],[],[]
        for im in [image_high_path]+positives_high+negtives_high:
            images_high.append(self.input_transform()(Image.open(im)))
        images_high=torch.stack(images_high)
        for im in [image_low_path]+positives_low+negtives_low:
            images_low_path.append(im)
            images_low.append(self.input_transform()(Image.open(im)))
        images_low=torch.stack(images_low)
        locations=torch.tensor(locs)
        return [images_high,images_low,images_low_path,locations]
    
    def input_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    reorganized = [dict() for _ in range(4)]    # images_high, images_low, images_low_path, locations for each model
    for batch_item in batch:
        for model, image_data in batch_item.items():
            # Distribute each type of image_data into the reorganized batch, where each type is still grouped into models
            for i in range(len(image_data)):
                if model in reorganized[i]:
                    reorganized[i][model].append(image_data[i])
                else:
                    reorganized[i][model] = [image_data[i]]
    # for model, low_paths in reorganized[2].items():    # TODO: undo comment
    #     reorganized[2][model] = [tuple(low_paths[batch_ind][path_ind] for batch_ind in range(len(low_paths))) for path_ind in range(len(low_paths[0]))]
    for data_ind in range(len(reorganized)):
        if data_ind == 2: continue
        for model in reorganized[data_ind]:
            reorganized[data_ind][model] = torch.stack(reorganized[data_ind][model])
    return reorganized


class Pitts250k_dataset(Dataset):
    
    def __init__(self, image_folder, resolution, models, gt, image_id, img_names=None, **config):
        self.gt=gt # (1000, 2)
        self.id=image_id # 1000
        self.nPosSample,self.nNegSample=config['nPosSample'],config['nNegSample'] # 1, 5
        self.high_resolution='raw'
        self.resolution = resolution
        self.input_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        neighbor_files = {model: h5py.File(join(image_folder, f"neighbors_{model}.h5"), "r") for model in models}
        self.__prepare_data(image_folder, models, img_names)
        for file in neighbor_files.values(): file.close()
        
    def __getitem__(self, index):
        image_data = {}
        for model, (high_image_set, low_image_set) in self.data[index].items():
            # concat image, postive image, and negative images
            image_high_path, positives_high, negtives_high = high_image_set
            image_low_path, positives_low, negtives_low = low_image_set
            images_high, images_low, images_low_path, locations = [], [], [], []
            for im in [image_high_path] + positives_high + negtives_high:
                images_high.append(self.input_transform(Image.open(im)))
            images_high=torch.stack(images_high)
            for im in [image_low_path] + positives_low + negtives_low:
                images_low_path.append(im)
                images_low.append(self.input_transform(Image.open(im)))
                locations.append(self.gt[self.id.index(int(basename(im).rstrip(".jpg")))]) # gt corresponding to the image path
            images_low=torch.stack(images_low)
            locations = torch.tensor(np.array(locations))
            image_data[model] = [images_high, images_low, images_low_path, locations]
            # images_high: (7, 3, 480, 640)
            # images_low: (7, 3, 180, 240)
            # images_low_path: 7 (list)
            # locations: (7, 2)
        return image_data
    
    def __len__(self):
        return len(self.data)
    
    def __prepare_data(self, image_folder, models, img_names):
        self.data, discarded_imgs = [], []
        dir_contents, pitch_list = listdir(image_folder), []
        for dir_entry in dir_contents:
            if isdir(join(image_folder, dir_entry)):
                pitch_list.append(dir_entry)
        for pitch in sorted(pitch_list):
            pitch_folder=join(image_folder,pitch)
            read_yaw_args = [(pitch, yaw, pitch_folder, models, image_folder, self.resolution, self.high_resolution,
                              self.nPosSample, self.nNegSample, img_names) for yaw in listdir(pitch_folder)]
            with multiprocessing.Pool() as pool:
                yaw_results = pool.starmap(Pitts250k_dataset.read_yaw_directory, read_yaw_args)
            for yaw_data, yaw_discard in yaw_results:
                self.data.extend(yaw_data)
                discarded_imgs.extend(yaw_discard)
        if len(discarded_imgs) > 0:
            print(f"Pitts250k dataset ({basename(image_folder)}) discarded {len(discarded_imgs)} for insufficient"
                  f"neighbors from at least one extractor.\n{discarded_imgs}")
    
    @staticmethod
    def read_yaw_directory(pitch, yaw, pitch_folder, models, image_folder, resolution, high_resolution, nPosSample, nNegSample, img_names):
        neighbor_files = {model: h5py.File(join(image_folder, f"neighbors_{model}.h5"), "r") for model in models}
        data, discarded_imgs = [], []
        images_root=join(pitch_folder, yaw)
        images_low_path=join(images_root, resolution) # '/scratch/lg3490/VPR4LQQ/logs/pitts250k/query/000/000/180p'
        images_high_path=join(images_root,'raw') # '/scratch/lg3490/VPR4LQQ/logs/pitts250k/query/000/000/raw'

        images_low=listdir(images_low_path)
        for image in images_low:
            if img_names is not None:
                orig_name = "_".join([image.rstrip(".jpg"), f"pitch{int(int(pitch) / 30 + 1)}", f"yaw{int(int(yaw) / 30 + 1)}"]) + ".jpg"
                if orig_name not in img_names:
                    continue
            name, image_data = f"{pitch}+{yaw}+{image}", {}
            for model, neighbor_file in neighbor_files.items():
                image_high_path=join(images_high_path,image) # '/scratch/lg3490/VPR4LQQ/logs/pitts250k/query/000/000/raw/006420.jpg'
                image_low_path=join(images_low_path,image)
                positives_high,negtives_high=[],[]
                positives_low,negtives_low=[],[]
                ind=0
                # key: how to get the neighbor thing in the new dataset
                positives_pool = neighbor_file[name]['positives'][:] #(20,)
                negtives_pool = neighbor_file[name]['negtives'][:] #(100,)
                while len(positives_high) < nPosSample:
                    pitch_,yaw_,name_=positives_pool[ind].decode('utf-8').split('+')
                    positive_high=join(image_folder, pitch_, yaw_, high_resolution, name_)
                    positive_low=join(image_folder, pitch_, yaw_ , resolution, name_)
                    if exists(positive_high) and exists(positive_low):
                        positives_high.append(positive_high)
                        positives_low.append(positive_low)
                    ind+=1
                    if ind==len(positives_pool)-1:
                        break
                ind=0
                while len(negtives_high)<nNegSample:
                    pitch_,yaw_,name_=negtives_pool[ind].decode('utf-8').split('+')
                    negtive_high=join(image_folder,pitch_,yaw_,high_resolution,name_)
                    negtive_low=join(image_folder,pitch_,yaw_,resolution,name_)
                    if exists(negtive_high) and exists(negtive_low):
                        negtives_high.append(negtive_high)
                        negtives_low.append(negtive_low)
                    ind+=1
                    if ind==len(negtives_pool)-1:
                        break
                if len(positives_high)==nPosSample and len(negtives_high)==nNegSample:
                    image_data[model] = [[image_high_path, positives_high, negtives_high], [image_low_path, positives_low, negtives_low]]
                # same pattern for low
            if len(image_data) == len(neighbor_files):
                data.append(image_data)
            else:
                discarded_imgs.append(name)
        for file in neighbor_files.values(): file.close()
        return data, discarded_imgs


def collate_fn_gsv(batch):
    batch_flattened = []    # Unprocessed batches have sublists for each place picked
    for place_imgs in batch:
        batch_flattened.extend(place_imgs)
    return collate_fn(batch_flattened)


class GSVCitiesDataset(Dataset):
    high_resolution = "raw"
    def __init__(self, images_path, data_path, resolution, cities, models, **config):
        self.images_path = images_path
        self.resolution = resolution
        self.models = models
        self.imgs_per_place = 4
        self.nPosSample, self.nNegSample = config["nPosSample"], config["nNegSample"]
        self.input_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.__prepare_data(images_path, data_path, cities, models, config["nPosSample"], config["nNegSample"])
        self.places_ids = pd.unique(self.img_data.index)
        self.neighbors_files = {city: {model: h5py.File(join(images_path, city, f"neighbors_{model}.h5"), "r") for model in models} for city in cities}
        print(f"GSV-Cities dataset loaded; images per place: {self.imgs_per_place}; cities: {cities}")
        
    def __getitem__(self, index):
        image_data = []
        place_id = self.places_ids[index]
        place = self.img_data.loc[place_id].sample(n=min(self.imgs_per_place, len(self.img_data.loc[place_id].index)))
        city = place.iloc[0]["city_id"]
        images_low_path, images_high_path = join(self.images_path, city, self.resolution), join(self.images_path, city, self.high_resolution)
        for _, row in place.iterrows():
            image_data_place = {}
            img_name = self.get_img_name(row, True)
            for model in self.models:
                images_high, high_descriptors, images_low, low_paths, locations = [], [], [], [], []
                neighbor_images_names = []
                for neighbor_type, neighbor_num in zip(["positives", "negtives"], [self.nPosSample, self.nNegSample]):
                    for neighbor_ind in range(neighbor_num):
                        neighbor_images_names.append(self.neighbors_files[city][model][img_name][neighbor_type][neighbor_ind].decode("utf-8"))
                for name in [img_name] + neighbor_images_names:
                    image_high_path, image_low_path = join(images_high_path, name), join(images_low_path, name)
                    images_high.append(self.input_transform(Image.open(image_high_path)))
                    # high_descriptors.append(torch.tensor(self.descriptor_files[city][model]["train"][basename(im)][:]))
                    # low_paths.append(image_low_path)    # TODO: undo comment
                    images_low.append(self.input_transform(Image.open(image_low_path)))
                    locations.append([row["lat"], row["lon"]])    # gt corresponding to the image path
                images_high = torch.stack(images_high)
                images_low = torch.stack(images_low)
                # high_descriptors = torch.stack(high_descriptors)
                locations = torch.tensor(convert_longlat_to_utm(np.array(locations)))
                image_data_place[model] = [images_high, images_low, low_paths, locations]
            image_data.append(image_data_place)
        return image_data
    
    def __len__(self):
        return len(self.places_ids)
    
    def close_neighbors_files(self):
        for city_files in self.neighbors_files.values():
            for model_file in city_files.values():
                model_file.close()
    
    def __prepare_data(self, images_path, data_path, cities, models, nPosSample, nNegSample):
        discarded_imgs = []
        cities_args = [(c_ind, cities, images_path, data_path, models, nPosSample, nNegSample) for c_ind in range(len(cities))]
        with multiprocessing.Pool() as pool:
            read_results = pool.starmap(GSVCitiesDataset.read_city_data, cities_args)
        self.img_data = pd.concat([read_result[0] for read_result in read_results], ignore_index=True).set_index("place_id")
        for read_result in read_results: discarded_imgs.extend(read_result[1])
        if len(discarded_imgs) > 0:
            print(f"GSV-Cities dataset discarded {len(discarded_imgs)} for insufficient"
                  f"neighbors from at least one extractor.\n{discarded_imgs}")
    
    @staticmethod
    def get_img_name(row, place_id_index=False):
        # given a row from the dataframe
        # return the corresponding image name

        city = row['city_id']
        
        # now remove the two digit we added to the id
        # they are superficially added to make ids different
        # for different cities
        pl_id = (row.name if place_id_index else row["place_id"]) % 10**5  #row.name is the index of the row, not to be confused with image name
        pl_id = str(pl_id).zfill(7)
        
        panoid = row['panoid']
        year = str(row['year']).zfill(4)
        month = str(row['month']).zfill(2)
        northdeg = str(row['northdeg']).zfill(3)
        lat, lon = str(row['lat']), str(row['lon'])
        name = city+'_'+pl_id+'_'+year+'_'+month+'_' + \
            northdeg+'_'+lat+'_'+lon+'_'+panoid+'.jpg'
        return name

    @staticmethod
    def read_city_data(c_ind, cities, images_path, data_path, models, nPosSample, nNegSample):
        discarded_imgs = []
        city_images = pd.read_csv(join(data_path, f"{cities[c_ind]}.csv"))
        neighbors_files = {model: h5py.File(join(images_path, cities[c_ind], f"neighbors_{model}.h5"), "r") for model in models}
        # For every image, load postives & negatives found by each model
        city_images["place_id"] = city_images["place_id"] + (c_ind * 10 ** 5)
        valid_rows = []
        for _, row in city_images.iterrows():
            img_name = GSVCitiesDataset.get_img_name(row)
            available_models = []
            for model, neighbor_file in neighbors_files.items():
                positives_pool = neighbor_file[img_name]["positives"]
                negatives_pool = neighbor_file[img_name]["negtives"]
                if len(positives_pool) >= nPosSample and len(negatives_pool) >= nNegSample:
                    available_models.append(model)
            valid_rows.append(len(available_models) == len(neighbors_files))
        valid_rows = pd.Series(valid_rows)
        valid_imgs = city_images[valid_rows].sample(frac=1)
        if len(valid_imgs.index) != len(city_images.index):
            discarded_imgs.extend(city_images[~valid_rows].apply(GSVCitiesDataset.get_img_name).to_list())
        for file in neighbors_files.values():
            file.close()
        return valid_imgs, discarded_imgs


def load_pitts250k_data(data, config, models):
    image_folder=data['image_folder']
    gt=data['utm'] # (1000, 2) eg. [585001.41335051, 4477058.99275442]
    image_id=data['id'] # 1000
    if "img_names" not in data: data["img_names"] = None

    dir_contents, pitch_list = listdir(image_folder), []
    for dir_entry in dir_contents:
        if isdir(join(image_folder, dir_entry)):
            pitch_list.append(dir_entry)
    
    for pitch in sorted(pitch_list):
        pitch_folder=join(image_folder,pitch)
        yaw_list=listdir(pitch_folder)
        for yaw in yaw_list:
            images_root=join(pitch_folder,yaw)
            resolution_list=sorted(listdir(images_root))
            break
        break

    dataset=[]
    dataconfig=config['data']
    if dataconfig['resolution']==-1:
        for resolution in tqdm(resolution_list,desc=f'loading data from {image_folder}',total=len(resolution_list)):
            if resolution!='raw':
                dataset.append(Pitts250k_dataset(image_folder, resolution, models, gt, image_id, data["img_names"], **config["triplet_loss"]))
    else:
        dataset.append(Pitts250k_dataset(image_folder, dataconfig["resolution"], models, gt, image_id, data["img_names"], **config["triplet_loss"]))

    return dataset


def load_gsv_cities_data(dataset_info, train_conf, cities, models):
    images_path, data_path = join(dataset_info["base_path"], "Images"), join(dataset_info["base_path"], "Dataframes")
    resolutions = []
    for dir_entry in listdir(join(images_path, cities[0])):
        if isdir(join(images_path, cities[0], dir_entry)):
            resolutions.append(dir_entry)
    resolutions = sorted(resolutions)

    datasets = []
    if train_conf["data"]["resolution"] == -1:
        for resolution in resolutions:
            if resolution != "raw":
                datasets.append(GSVCitiesDataset(images_path, data_path, resolution, cities, models, **train_conf["triplet_loss"]))
    else:
        datasets.append(GSVCitiesDataset(images_path, data_path, train_conf["data"]["resolution"], cities, models, **train_conf["triplet_loss"]))
    return datasets


def load_data(image_folder,**config):
    pitch_list=listdir(image_folder)
    pitch_list.remove('global_descriptor.h5')
    pitch_list.remove('neighbors.h5')
    
    for pitch in sorted(pitch_list):
        pitch_folder=join(image_folder,pitch)
        yaw_list=listdir(pitch_folder)
        for yaw in yaw_list:
            images_root=join(pitch_folder,yaw,'images')
            resolution_list=sorted(listdir(images_root))
            qp_list=sorted(listdir(join(images_root,resolution_list[0])))
            break
        break

    dataset=[]
    dataconfig=config['data']
    if dataconfig['resolution']==-1:
        for i,resolution in tqdm(enumerate(resolution_list),desc=f'loading data from {image_folder}',total=len(resolution_list)):
            if dataconfig['qp']==-1:
                if i==len(resolution_list)-1:
                    qp_list.remove('raw')
                for qp in qp_list:
                    dataset.append(UNav_dataset(image_folder,resolution,qp,pitch_list,yaw_list,**config['triplet_loss']))
            else:
                dataset.append(UNav_dataset(image_folder,resolution,dataconfig['qp'],pitch_list,yaw_list,**config['triplet_loss']))
    else:
        if dataconfig['qp']==-1:
            for qp in qp_list:
                dataset.append(UNav_dataset(image_folder,dataconfig['resolution'],qp,pitch_list,yaw_list,**config['triplet_loss']))
        else:
            dataset.append(UNav_dataset(image_folder,dataconfig['resolution'],dataconfig['qp'],pitch_list,yaw_list,**config['triplet_loss']))

    return dataset

def load_test_data(image_folder,**config):
    pitch_list=listdir(image_folder)
    pitch_list.remove('global_descriptor.h5')
    pitch_list.remove('neighbors.h5')
    for pitch in sorted(pitch_list):
        pitch_folder=join(image_folder,pitch)
        yaw_list=listdir(pitch_folder)
        for yaw in yaw_list:
            images_root=join(pitch_folder,yaw,'images')
            resolution_list=sorted(listdir(images_root))
            qp_list=sorted(listdir(join(images_root,resolution_list[0])))
            break
        break
    dataset={}
    for i,resolution in tqdm(enumerate(resolution_list),desc=f'loading data from {image_folder}',total=len(resolution_list)):
        if resolution not in dataset:
            dataset[resolution]={}
        for qp in qp_list:
            dataset[resolution][qp]=UNav_dataset(image_folder,resolution,qp,pitch_list,yaw_list,**config['triplet_loss'])
    return dataset

def data_link(root,name,scales,data_list):
    data={}
    for d in data_list:
        key=d.split('_')[0]
        if key in scales:
            data[d]={'image_folder':join(root,'logs',name,d),'utm_file':join(root,'data',name,'utm',d+'.json'),'scale':scales[key]}
    return data

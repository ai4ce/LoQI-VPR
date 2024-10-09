import argparse
import random
import shutil
import cv2
import h5py
from os import listdir, mkdir, rename
from os.path import basename, exists, join, splitext
import multiprocessing
import numpy as np
from PIL import Image
import resource
from sklearn.neighbors import NearestNeighbors
import torch
from torchvision import transforms
from tqdm import tqdm
import utm
import yaml

from feature.Global_Extractors import GlobalExtractors


TRAIN_CITIES = [
    "Bangkok",
    "BuenosAires",
    "LosAngeles",
    "MexicoCity",
    "OSL",
    "Rome",
    "Barcelona",
    "Chicago",
    "Madrid",
    "Miami",
    "Phoenix",
    "TRT",
    "Boston",
    "Lisbon",
    "Medellin",
    "Minneapolis",
    "PRG",
    "WashingtonDC",
    "Brussels",
    "London",
    "Melbourne",
    "Osaka",
    "PRS"
]

CITIES_300x400 = [    # Images for these cities all have height 300 and width 400
    "OSL",
    "Bangkok",
    "Osaka",
    "WashingtonDC",
    "TRT",
    "MexicoCity",
    "Medellin",
    "LosAngeles",
    "PRG",
    "Barcelona",
    "BuenosAires",
    "PRS"
]

CITIES_480x640 = [    # Images for these cities all have height 480 and width 640
    "Boston",
    "Brussels",
    "Phoenix",
    "London",
    "Lisbon",
    "Melbourne",
    "Minneapolis",
    "Miami",
    "Rome",
    "Chicago",
    "Madrid"
]


input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def convert_longlat_to_utm(longlat_coords):    # Coordinates would be closer than reality for large distances (eg. few hundred kms)
    def convert_coord(coord):
        x, y, zone, _ = utm.from_latlon(coord[0], coord[1])
        return np.array([x, y])
    return np.apply_along_axis(convert_coord, 1, longlat_coords)


def extract_descriptors(image_path, city, model, global_extractors):
    hfile_path = join(image_path, city, f"global_descriptor_{model}.h5")
    hfile = h5py.File(hfile_path, "a")
    # Retrieve already processed images, if any
    if "train" in hfile:
        existing_imgs = set(hfile["train"])
        grp = hfile["train"]
    else:
        existing_imgs = set()
        grp = hfile.create_group("train")

    # Ignore already processed images
    images_high_path = join(image_path, city, "raw")
    image_high_names = set(listdir(images_high_path))
    images_to_add = image_high_names.difference(existing_imgs)

    if len(images_to_add) == 0:
        print(f"{city} {basename(hfile_path)} already contains all images.")
        hfile.close()
        return
    for i, im in tqdm(enumerate(images_to_add), desc=f"Extracting {city} {basename(hfile_path)}", total=len(images_to_add)):
        image = Image.open(join(images_high_path, im))
        if image.mode != "RGB": image = image.convert("RGB")
        image = input_transform(image)
        # If 200 images read, no more images to read, or image resolution changed, compute descriptors
        if i % 200 == 0 or i == len(images_to_add) - 1 or (len(images_list) > 0 and images_list[-1].shape != image.shape):
            if i > 0:
                batched_imgs = torch.stack(images_list)
                encodings, descriptors = global_extractors(model, batched_imgs)
                for name, descriptor in zip(images_name, descriptors.cpu()):
                    grp.create_dataset(name, data=descriptor)
                del batched_imgs, descriptors
                torch.cuda.empty_cache()
            images_list, images_name = [], []
        images_list.append(image.to(device))
        images_name.append(im)
    if len(images_list) == 1: grp.create_dataset(im, data=global_extractors(model, image.to(device).unsqueeze(0))[1].cpu().squeeze(0))
    hfile.close()


def neighbor_matcher(batch_idx, batch_size, descriptors, names, indices_to_add, nontrivial_positives, potential_negatives, nPosSample):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(names))
    if len(indices_to_add.intersection(range(start_idx, end_idx))) == 0: return None
    descriptor_batch = descriptors[start_idx:end_idx]
    # Compute the similarity matrix for the batch against all descriptors
    sim_batch = torch.einsum("id,jd->ij", descriptor_batch, descriptors).float()
    for i in range(descriptor_batch.shape[0]):
        sim_batch[i, start_idx + i] = 0
    
    matched_info = []
    # Between start and end index, keep only uncalculated indices (ie. the corresponding image isn't in neighbors file)
    for i in [ind - start_idx for ind in indices_to_add.intersection(range(start_idx, end_idx))]:
        name = names[start_idx + i]
        sim = sim_batch[i]
        feature_closed = torch.topk(sim, descriptors.size(0), dim=0).indices.numpy()
        
        physical_closed = set([names[ind] for ind in nontrivial_positives[start_idx + i]])
        
        negatives = [names[ind] for ind in potential_negatives[start_idx + i]]
        negatives = random.sample(negatives, 100)

        positives = []
        ind = 0
        while len(positives) < nPosSample * 20:
            key = names[feature_closed[ind]]
            if key in physical_closed:
                positives.append(names[feature_closed[ind]])
            ind += 1
            if ind >= len(feature_closed):
                break

        if len(positives) > 0:
            matched_info.append((name, positives, negatives))
    return matched_info


def find_neighbors(images_path, city, model, all_coords, global_descriptor_dim, posDistThr, nonTrivPosDistSqThr, nPosSample):
    print(f"{city} memory start of neighbors: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
    hfile_neighbor_path = join(images_path, city, f"neighbors_{model}.h5")
    hfile_path = join(images_path, city, f"global_descriptor_{model}.h5")
    hfile = h5py.File(hfile_path, "r")
    
    names = []
    descriptors = np.empty((len(hfile["train"]), global_descriptor_dim))
    city_coords = np.empty((len(hfile["train"]), 2), dtype=np.longdouble)

    for i, (img_name, img_feat) in tqdm(enumerate(hfile["train"].items()), desc=f"Loading {city} {basename(hfile_path)}", total=len(hfile["train"])):
        names.append(img_name)
        city_coords[i, :] = all_coords[img_name]
        descriptors[i, :] = img_feat.__array__()
    hfile.close()

    # Open the HDF5 file for storing neighbors
    hfile = h5py.File(hfile_neighbor_path, "a")
    existing_indices = set(names.index(name) for name in set(hfile))
    indices_to_add = set(range(len(names))).difference(existing_indices)

    if len(indices_to_add) == 0:
        print(f"{city} {basename(hfile_neighbor_path)} already contains all images.")
        hfile.close()
        return

    locations = convert_longlat_to_utm(city_coords)

    knn = NearestNeighbors(n_jobs=-1)
    knn.fit(locations)
    nontrivial_positives = list(knn.radius_neighbors(locations,
                                                     radius=nonTrivPosDistSqThr,
                                                     return_distance=False))

    potential_positives = knn.radius_neighbors(locations,
                                               radius=posDistThr,
                                               return_distance=False)
    potential_negatives = []
    for pos in potential_positives:
        potential_negatives.append(np.setdiff1d(np.arange(len(locations)),
                                                pos, assume_unique=True))

    descriptors = torch.from_numpy(descriptors)
    batch_size = 1000  # or any other batch size you find suitable

    # Iterate through batches to find neighbors
    num_batches = int(np.ceil(len(names)/batch_size))
    matcher_args = [(batch_idx, batch_size, descriptors, names, indices_to_add, nontrivial_positives, potential_negatives, nPosSample)
                    for batch_idx in range(num_batches)]
    for arg in tqdm(matcher_args, desc=f"{city} {basename(hfile_neighbor_path)}"):
        matched_info = neighbor_matcher(*arg)
        if matched_info is not None:
            for img_neighbor_info in matched_info:
                grp = hfile.create_group(img_neighbor_info[0])
                grp.create_dataset("positives", data=img_neighbor_info[1])
                grp.create_dataset("negtives", data=img_neighbor_info[2])
    hfile.close()


def resize_images(images_path, city, compressions):
    city_path = join(images_path, city)
    # Move any unclassifed images under database or query to the "raw" resolution folder
    images = [image for image in listdir(city_path) if splitext(image)[1] == ".jpg"]
    raw_folder = join(city_path, "raw")
    if not exists(raw_folder):
        mkdir(raw_folder)
    for image in images:
        rename(join(city_path, image), join(raw_folder, image))
    # Ensure all other resolutions have a cooresponding folder
    for comp_name in compressions:
        image_folder = join(city_path, comp_name)
        if not exists(image_folder):
            mkdir(image_folder)
    
    raw_images = listdir(raw_folder)
    # Resize and copy all images moved to raw folder into corresponding resolution folders
    for image_name in tqdm(raw_images, desc=f'Processing {city} images'):
        raw_image_path = join(raw_folder, image_name)
        comps_todo = []
        for comp_name in compressions:
            if not exists(join(city_path, comp_name, image_name)):
                comps_todo.append(comp_name)
        if len(comps_todo) > 0:
            image = Image.open(raw_image_path)
            for comp_name in comps_todo:
                image.save(join(city_path, comp_name, image_name), "JPEG", quality=compressions[comp_name])
            image.close()


def read_gt(images_path):
    # Initialize lists to store UTM coordinates and panorama IDs
    longlat_coords, image_names = [], []
    for city in tqdm(listdir(images_path), desc=f"Reading image groundtruth"):
        for image_name in listdir(join(images_path, city, "raw")):
            parts = image_name.split("_")
            try:
                # Assume the x and y coordinates of image ground truth is 1st and 2nd component
                utm_east, utm_north = float(parts[5].strip()), float(parts[6].strip())
                
                # Append the extracted information to the lists
                longlat_coords.append([utm_east, utm_north])
                image_names.append(image_name)
            except (ValueError, AssertionError):
                # If conversion to float fails, skip this file
                print(f"Skipping file due to invalid format: {image_name}")
    return np.array(longlat_coords), image_names


def main(config):
    config["vpr"]["global_extractor"]["CricaVPR"]["cuda"] = False    # TODO: remove
    global_extractors = GlobalExtractors(config["root"], config["vpr"]["global_extractor"], pipeline=False)
    dataset_source, dataset_dest = join(config["root"], "data", "third_party", "GSV-Cities"), join(config["root"], "logs", "GSV-Cities")
    if not exists(dataset_dest):
        shutil.copytree(dataset_source, dataset_dest)
    images_path = join(dataset_dest, "Images")
    resize_args = [(images_path, city, config["data"]["compression"]["compressions"]) for city in CITIES_300x400]
    with multiprocessing.Pool() as pool:
        pool.starmap(resize_images, resize_args)
    longlat_coords, image_names = read_gt(images_path)
    groundtruth = dict(zip(image_names, longlat_coords))

    posDistThr = config["train"]["triplet_loss"]["posDistThr"]
    nonTrivPosDistSqThr = config["train"]["triplet_loss"]["nonTrivPosDistSqThr"]
    nPosSample = config["train"]["triplet_loss"]["nPosSample"]

    arguments = [[(images_path, city, model, global_extractors),
                 (images_path, city, model, groundtruth, global_extractors.feature_length(model), posDistThr, nonTrivPosDistSqThr, nPosSample)]
                 for city in TRAIN_CITIES for model in global_extractors.models]
    # Separate arguments column-wise to feed feature extraction and positive / negative pair finding
    descriptor_args, neighbor_args = zip(*arguments)
    for arg in descriptor_args:
        extract_descriptors(*arg)
    for arg in neighbor_args:
        find_neighbors(*arg)


if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="../configs/trainer.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    main(config)
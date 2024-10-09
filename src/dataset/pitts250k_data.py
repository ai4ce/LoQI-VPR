from os.path import join,exists,basename
from os import listdir,makedirs
import multiprocessing
import numpy as np
from sklearn.neighbors import NearestNeighbors
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import h5py
from feature.Global_Extractors import GlobalExtractors
import argparse
import yaml
import json
import random
import scipy.io
import shutil

def get_gt(aligner_path):
    with open(aligner_path, "r") as f:
        keyframes = json.load(f)['keyframes']
    key= {}
    for id, point in keyframes.items():
        t_mp=point['trans']
        rot=point['rot']
        key[str(int(id)-1).zfill(5)] = [t_mp[0], t_mp[1], rot]
    return key

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ])

def extract_descriptors(image_folder, model, global_extractors):
    hfile_path=join(image_folder, f'global_descriptor_{model}.h5')
    pitch_list=['000','030']
    yaw_list=[str(i*30).zfill(3) for i in range(12)]

    hfile, grp_name = h5py.File(hfile_path, 'a'), basename(image_folder)
    # Retrieve already processed images, if any
    if grp_name in hfile:
        existing_imgs = set(hfile[grp_name])
        grp = hfile[grp_name]
    else:
        existing_imgs = set()
        grp = hfile.create_group(basename(image_folder))
    
    image_high_names = []
    for pitch in sorted(pitch_list):
        pitch_folder = join(image_folder, pitch)
        for yaw in sorted(yaw_list):
            images_high_path = join(pitch_folder, yaw, 'raw')
            curr_angle_imgs = set(sorted(listdir(images_high_path)))
            image_high_names.extend([f"{pitch}+{yaw}+{orig_name}" for orig_name in curr_angle_imgs])
    image_high_names = set(image_high_names)
    images_to_add = image_high_names.difference(existing_imgs)

    if len(images_to_add) == 0:
        print(f"{basename(image_folder)} {basename(hfile_path)} already contains all images.")
        hfile.close()
        return
    for i, img_identifier in tqdm(enumerate(images_to_add), desc=f"{basename(image_folder)} {basename(hfile_path)}", total=len(images_to_add)):
        pitch, yaw, orig_name = img_identifier.split("+")
        image_high_path = join(image_folder, pitch, yaw, "raw", orig_name)
        if i % 100 == 0 or i == len(images_to_add) - 1:
            if i>0:
                image_=torch.stack(images_list)
                with torch.no_grad():
                    encodings, global_descr = global_extractors(model, image_)
                for name, descriptor in zip(images_name, global_descr.cpu()):
                    grp.create_dataset(name, data=descriptor)
                del image_, global_descr
                torch.cuda.empty_cache()
            images_list, images_name = [], []
        image = input_transform()(Image.open(image_high_path))
        images_list.append(image.to(device))
        images_name.append(img_identifier)
    if len(images_list) == 1: grp.create_dataset(img_identifier, data=global_extractors(model, image.to(device).unsqueeze(0))[1].squeeze(0).cpu())
    hfile.close()

def find_neighbors(image_folder, gt, global_descriptor_dim, model, posDistThr, nonTrivPosDistSqThr, nPosSample, query=False):
    if query:
        gt_name=join(config['root'],'logs/pitts250k/pittsburgh_queryID_1000.mat')
        name_id = scipy.io.loadmat(gt_name)['query_id'][0]
        name_id=[int(i)-1 for i in name_id]
    else:
        name_id=[i for i,_ in enumerate(gt)]

    pitch_list=['000','030']
    yaw_list=[str(i*30).zfill(3) for i in range(12)]

    hfile_neighbor_path = join(image_folder, f"neighbors_{model}.h5")
    hfile_path = join(image_folder, f"global_descriptor_{model}.h5")
    hfile = h5py.File(hfile_path, 'r')
    
    names=[]
    descriptors=np.empty((len(hfile[basename(image_folder)]),global_descriptor_dim))
    locations=np.empty((len(hfile[basename(image_folder)]),2))

    for i,(k,v) in tqdm(enumerate(hfile[basename(image_folder)].items()),desc='load data',total=len(hfile[basename(image_folder)])):
        names.append(k)
        pitch,yaw,name=k.replace('.jpg','').split('+')
        locations[i,:]=gt[name_id.index(int(name))]
        descriptors[i,:]=v.__array__()
    hfile.close()

    knn = NearestNeighbors(n_jobs=-1)
    knn.fit(gt)
    nontrivial_positives = list(knn.radius_neighbors(gt,
            radius=nonTrivPosDistSqThr, 
            return_distance=False))

    potential_positives = knn.radius_neighbors(gt,
            radius=posDistThr, 
            return_distance=False)
    potential_negatives = []
    for pos in potential_positives:
        potential_negatives.append(np.setdiff1d(np.arange(len(gt)),
            pos, assume_unique=True))

    descriptors=torch.from_numpy(descriptors)
    batch_size = 1000  # or any other batch size you find suitable

    # Open the HDF5 file
    hfile = h5py.File(hfile_neighbor_path, 'a')
    existing_indices = set(names.index(name) for name in set(hfile))
    indices_to_add = set(range(len(names))).difference(existing_indices)

    # Iterate through batches
    num_batches = int(np.ceil(len(names)/batch_size))

    if len(indices_to_add) == 0:
        print(f"{basename(image_folder)} {basename(hfile_neighbor_path)} already contains all images.")
        hfile.close()
        return
    for batch_idx in tqdm(range(num_batches), desc='find neighbor'):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(names))
        if len(indices_to_add.intersection(range(start_idx, end_idx))) == 0: continue
        descriptor_batch = descriptors[start_idx:end_idx]
        # Compute the similarity matrix for the batch against all descriptors
        sim_batch = torch.einsum('id,jd->ij', descriptor_batch, descriptors).float()
        for i in range(descriptor_batch.shape[0]):
            sim_batch[i, start_idx + i] = 0
        
        for i in [ind - start_idx for ind in indices_to_add.intersection(range(start_idx, end_idx))]:
            name = names[start_idx + i]
            sim = sim_batch[i]
            feature_closed = torch.topk(sim, descriptors.size(0), dim=0).indices.numpy()
            
            key = name.replace('.jpg', '').split('+')[-1]
            physical_closed = set([str(name_id[ind]).zfill(6) for ind in nontrivial_positives[name_id.index(int(key))]])
            
            negtives_ = [str(name_id[ind]).zfill(6) for ind in potential_negatives[name_id.index(int(key))]]
            negtives = []
            for pitch in pitch_list:
                for yaw in yaw_list:
                    for neg in negtives_:
                        negtives.append(f'{pitch}+{yaw}+{neg}.jpg')
            negtives = random.sample(negtives, 100)

            positives = []
            ind = 0
            while len(positives) < nPosSample * 20:
                key = names[feature_closed[ind]].replace('.jpg', '').split('+')[-1]
                if key in physical_closed:
                    positives.append(names[feature_closed[ind]])
                ind += 1
                if ind >= len(feature_closed):
                    break

            if len(positives) > 0:
                grp = hfile.create_group(name)
                grp.create_dataset('positives', data=positives)
                grp.create_dataset('negtives', data=negtives)

    hfile.close()

def move_folder_images(input_folder, folder, train_img_names, test_img_names, val_database_names, output_folder, compressions, resolutions):
    images=listdir(join(input_folder,folder))
    images.remove('tar')
    for image in tqdm(images,desc=f'process {folder}',total=len(images)):
        name,pitch,yaw=image.replace('.jpg','').split('_')
        pathname = join(folder, image)
        type = None
        pitch=str((int(pitch.replace('pitch',''))-1)*30).zfill(3)
        yaw=str((int(yaw.replace('yaw',''))-1)*30).zfill(3)
        if pathname in train_img_names:
            type='train'
        elif pathname in test_img_names:
            type='valid'
        elif pathname in val_database_names:
            type='database'
        if type is None: continue
        input_path=join(input_folder,folder,image)
        output_path=join(output_folder,type,pitch,yaw)
        # shutil.copy(input_path,join(output_path,'raw',name+'.jpg'))    # TODO: remove comment
        comps_todo = []
        for comp_name in compressions:
            if not exists(join(output_path, comp_name)):
                makedirs(join(output_path, comp_name))
            if not exists(join(output_path, comp_name, name + ".jpg")):
                comps_todo.append(comp_name)
        if len(comps_todo) > 0:
            image = Image.open(input_path)
            for comp_name in comps_todo:
                image.save(join(output_path, comp_name, name + ".jpg"), "JPEG", quality=compressions[comp_name])
            image.close()
        image_=cv2.imread(input_path)
        for resolution,newsize in resolutions.items():
            image_new=cv2.resize(image_,newsize)
            cv2.imwrite(join(output_path,resolution,name+'.jpg'),image_new)

def process_data(root, split_info_path, compressions, resolutions):
    output_folder=join(root,'logs','pitts250k')
    types=['train','valid','database','query']
    pitch_list=['000','030']
    yaw_list=[str(i*30).zfill(3) for i in range(12)]
    resolutions={'raw':[640,480],'180p':[240,180],'360p':[480,360],'240p':[320,240]}
    for t in types:
        outf=join(output_folder,t)
        for PITCH in pitch_list:
            pitch_folder=join(outf,PITCH)
            for YAW in yaw_list:
                yaw_folder=join(pitch_folder,YAW)
                for k in list(resolutions.keys()):
                    image_folder=join(yaw_folder,k)
                    if not exists(image_folder):
                        makedirs(image_folder)
    resolutions.pop('raw')

    database_folders=[str(i).zfill(3) for i in range(11)]

    train_images = scipy.io.loadmat(join(split_info_path, "pitts250k_train.mat"))["dbStruct"].item()[1]
    train_img_names = [image[0].item() for image in train_images]
    val_struct = scipy.io.loadmat(join(split_info_path, "pitts250k_val.mat"))["dbStruct"].item()
    val_database_names = [image[0].item() for image in val_struct[1]]
    val_query_names = [basename(image[0].item()) for image in val_struct[3]]
    test_images = scipy.io.loadmat(join(split_info_path, "pitts250k_test.mat"))["dbStruct"].item()[1]
    test_img_names = [image[0].item() for image in test_images]

    input_folder=join(root,'data','third_party','pitts250k')
    move_images_args = [(input_folder, folder, train_img_names, test_img_names, val_database_names, output_folder, compressions, resolutions) for folder in database_folders]
    with multiprocessing.Pool() as pool:
        pool.starmap(move_folder_images, move_images_args)

    query_folder=join(root,'data','third_party','pitts250k','queries_real')
    images=listdir(query_folder)
    images.remove('tar')
    for image in images:
        name,pitch,yaw=image.replace('.jpg','').split('_')
        pitch=str((int(pitch.replace('pitch',''))-1)*30).zfill(3)
        yaw=str((int(yaw.replace('yaw',''))-1)*30).zfill(3)
        if image in val_query_names:
            type='query'
            input_path=join(query_folder,image)
            output_path=join(output_folder,type,pitch,yaw)
            comps_todo = []
            for comp_name in compressions:
                if not exists(join(output_path, comp_name)):
                    makedirs(join(output_path, comp_name))
                if not exists(join(output_path, comp_name, name + ".jpg")):
                    comps_todo.append(comp_name)
            if len(comps_todo) > 0:
                image = Image.open(input_path)
                for comp_name in comps_todo:
                    image.save(join(output_path, comp_name, name + ".jpg"), "JPEG", quality=compressions[comp_name])
                image.close()
            image_=cv2.imread(input_path)
            for resolution,newsize in resolutions.items():
                image_new=cv2.resize(image_,newsize)
                cv2.imwrite(join(output_path,resolution,name+'.jpg'),image_new)

def main(configs):
    root=configs['root']
    global_extractors = GlobalExtractors(configs["root"], configs["vpr"]["global_extractor"], pipeline=False)

    gt_path=join(root,'logs/pitts250k/pittsburgh_database_10586_utm.mat')
    database_gt = scipy.io.loadmat(gt_path)['Cdb'].T
    
    gt_path=join(root,'logs/pitts250k/pittsburgh_query_1000_utm.mat')
    split_info_path = join(root, "data/third_party/pitts250k/netvlad_v100_datasets/tar/datasets")
    query_gt = scipy.io.loadmat(gt_path)['Cq'].T
    if not exists(join(root,'logs','pitts250k')):
        process_data(root, split_info_path, config["data"]["compression"]["compressions"], config["data"]["compression"]["resolution"])

    posDistThr=configs['train']['triplet_loss']['posDistThr']
    nonTrivPosDistSqThr=configs['train']['triplet_loss']['nonTrivPosDistSqThr']
    nPosSample=configs['train']['triplet_loss']['nPosSample']

    for image_folder in ['query','database','train','valid']:
        print(f'======================Processing {image_folder}')
        if image_folder=='query':
            gt=query_gt
        else:
            gt=database_gt
        for model in global_extractors.models:
            extract_descriptors(join(root,'logs','pitts250k',image_folder), model, global_extractors)
            find_neighbors(join(root,'logs','pitts250k',image_folder), gt, global_extractors.feature_length(model), model,
                           posDistThr, nonTrivPosDistSqThr, nPosSample, query=image_folder=="query")

            
if __name__=='__main__':
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../configs/trainer.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    main(config)
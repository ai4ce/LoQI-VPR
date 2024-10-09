import argparse
import math
import os
from os.path import basename, join
import pandas as pd
import random
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import torch
import h5py

from dataset.Data_Control import BatchSampler
from dataset.test_dataset import TestDataset, collate_fn
from feature.Global_Extractors import GlobalExtractors


class VPRTester:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sheet_start_column = "D"
    sheet_start_row = 3

    def __init__(self, root, data_conf, data_folders, vpr_conf, train_conf, model_IO, query_batch):
        self.extrs_pretrained = GlobalExtractors(root, vpr_conf["global_extractor"], data_parallel=train_conf["multiGPU"])
        # Assemble logs and weights folder name using training data name and loss and data configurations
        train_data = train_conf["data"]["name"]
        if train_conf["finetuned"]:
            log_suffix = join(f"{train_data}_finetuned", "{}")
        else:
            log_suffix = join(train_data + ("_distill" if train_conf["loss"]["distill"] else "") + ("_vlad" if train_conf["loss"]["vlad"] else "") \
                            + ("_triplet" if train_conf["loss"]["triplet"] else ""), "{}",
                            str(train_conf["data"]["resolution"]) + "_" + str(train_conf["data"]["qp"]) + "_" + str(train_conf["lr"]))

        self.retrieval_save_path, self.retrieval_database = {}, {}        
        for model_name in self.extrs_pretrained.models:
            ckpt_path = join(root, model_IO["weights_path"], log_suffix.format(model_name))
            vpr_conf["global_extractor"][model_name]["ckpt_path"] = ckpt_path
            self.retrieval_save_path[model_name] = join(ckpt_path, f"{data_conf['name']}_retrieval_database.pkl")
            self.retrieval_database[model_name] = {col_name: [] for col_name in ["query_path", "version", "success", "db_images"]}
        # Load the trained model, picking "model_best.pth" as weights via type="pipeline"
        self.extrs_trained = GlobalExtractors(root, vpr_conf["global_extractor"], pipeline=True, data_parallel=train_conf["multiGPU"])
        
        log_suffix = join(*[folder_name for folder_name in log_suffix.split(os.path.sep) if folder_name != "{}"])
        self.thresholds = torch.tensor(vpr_conf["threshold"])
        self.topk_nodes = torch.tensor(vpr_conf["topk"])
        self.batch_size = train_conf["batch_size"]
        
        self.train_conf = train_conf
        pretrained_suffix, trained_suffix = join("precomputed_descriptors", "pretrained"), join("precomputed_descriptors", log_suffix)
        self.database_images_set = self.load_database_images(data_folders, pretrained_suffix, trained_suffix, data_conf)
        self.query_images_set = TestDataset(data_folders["query"], data_conf["test_res"], train_conf["triplet_loss"])
        self.query_batch = query_batch
        self.dataset_ind = get_data_index(data_conf["name"])
        self.data_conf = data_conf


    def load_database_images(self, data_folders, pretrained_suffix, trained_suffix, data_conf):
        image_folder, use_trained_descs = data_folders["database"], data_conf["use_trained_descs"]
        desc_file_suffix = f"{{}}.h5" if data_conf["test_res"] == "raw" else f"{{}}_{data_conf['test_res']}.h5"

        # Create descriptor file handles and empty descriptor tensors
        g_desc_files = {f"pretrained_{model}": h5py.File(join(image_folder, pretrained_suffix, f"global_descriptor_{desc_file_suffix.format(model)}"), 'r') for model in self.extrs_pretrained.models}
        if use_trained_descs:
            g_desc_files.update({model: h5py.File(join(image_folder, trained_suffix, f"global_descriptor_{desc_file_suffix.format(model)}"), 'r') for model in self.extrs_trained.models})
        num_images = len(g_desc_files[list(g_desc_files.keys())[0]][basename(image_folder)])
        assert all([num_images == len(file[basename(image_folder)]) for file in g_desc_files.values()]), f"Database global descriptors for {image_folder} have different lengths (first file has length {num_images})."
        descriptors = {model: torch.empty((num_images, self.extrs_trained.feature_length(model))) for model in self.extrs_trained.models}
        for model, desc_mat in descriptors.items(): descriptors[model] = {"pretrained": desc_mat, "trained": desc_mat.clone() if use_trained_descs else desc_mat}
        locations = torch.empty((num_images, 2))
        image_paths = []
        for file_ind, (model, g_desc_file) in enumerate(g_desc_files.items()):
            model_name, version = model.removeprefix("pretrained_"), "pretrained" if model.startswith("pretrained_") else "trained"
            for i, (name, d) in enumerate(tqdm(g_desc_file[basename(image_folder)].items(), desc=f"Reading database info for {model}")):
                if file_ind == 0:
                    splitted = name.split('@')
                    utm_east_str, utm_north_str = splitted[1], splitted[2]
                    image_path = join(image_folder, data_conf["test_res"], name)
                    image_paths.append(image_path)
                    locations[i] = torch.tensor([float(utm_east_str), float(utm_north_str)], dtype=torch.float32)
                descriptors[model_name][version][i] = torch.tensor(d.__array__())
            g_desc_file.close()

        return {'images_path': image_paths, 'descriptors': descriptors, 'locations': locations}
    
    def vpr_examing(self, query_desc, database_desc, query_loc, database_loc, query_paths, database_paths, model_name, version):
        query_desc, database_desc, query_loc, database_loc = query_desc.to(self.device), database_desc.to(self.device), query_loc.float().to(self.device), database_loc.float().to(self.device)
        sim = torch.einsum('id,jd->ij', query_desc, database_desc)
        topk_ = torch.topk(sim, self.topk_nodes[-1], dim=1)
        topk=topk_.indices
        success_num = torch.zeros((self.thresholds.shape[0],self.topk_nodes.shape[0]))
        for i, index in enumerate(topk):
            qloc = query_loc[i].unsqueeze(0)[:, :2]
            dloc = database_loc[index][:, :2]

            distance = torch.cdist(qloc, dloc, p=2).squeeze(0)

            matched = False
            for thre_ind,thre in enumerate(self.thresholds):
                if not matched:
                    if thre_ind==0:
                        ind = torch.where(distance < thre)
                    else:
                        ind=torch.where((distance < thre)&(distance>=self.thresholds[thre_ind-1]))
                    index_match=index[ind]
                    if len(index_match) > 0:
                        maskA = torch.zeros((self.thresholds.shape[0], self.topk_nodes.shape[0]), dtype=torch.bool)
                        maskB = torch.zeros((self.thresholds.shape[0], self.topk_nodes.shape[0]), dtype=torch.bool)
                        maskA[self.thresholds >= thre, :] = True
                        maskB[:, self.topk_nodes >= torch.min(ind[0]).item()] = True
                        mask = (maskA & maskB)
                        success_num[mask] += 1
                        matched = True

            query_path = query_paths[i][1]    # [0] gets high resolution; [1] gets "low" resolution (or compressed)
            self.retrieval_database[model_name]["query_path"].append(query_path)
            self.retrieval_database[model_name]["version"].append(version)
            self.retrieval_database[model_name]["success"].append(matched)
            db_paths_of_query = [database_paths[success_db_ind] for success_db_ind in (index_match if matched else index)]
            self.retrieval_database[model_name]["db_images"].append(db_paths_of_query)

            del qloc,dloc,distance
        del sim,topk,query_desc,database_desc, query_loc, database_loc
        torch.cuda.empty_cache()
        return success_num

    def validation(self):
        print(f"Start vpr examining ...")
        
        # All query images are evaluated for VPR recall
        batch_sampler = BatchSampler([list(range(len(self.query_images_set)))], self.query_batch)
        data_loader = DataLoader(self.query_images_set, collate_fn=collate_fn, num_workers=self.train_conf['num_worker'], pin_memory=True, batch_sampler=batch_sampler)

        recall_score = {model: {version: torch.zeros((self.thresholds.shape[0],self.topk_nodes.shape[0])) for version in ["pretrained", "trained"]} for model in self.extrs_trained.models}
        with torch.no_grad():
            for images_high, images_low, locations, paths in tqdm(data_loader, total=math.ceil(len(self.query_images_set) / self.query_batch)):
                # For each image batch, apply pretrained and trained versions of each extractor and calculate separate recall rates
                locations = locations[:, 0, :]
                query_paths = [path_per_batch[0] for path_per_batch in paths]
                features_low, vectors_low = self._compute_image_descriptors(images_low, recall_score.keys())
                for model, vector_low in vectors_low.items():
                    for version in ["pretrained", "trained"]:
                        recall_score[model][version] += self.vpr_examing(vector_low[version], self.database_images_set["descriptors"][model][version], locations, self.database_images_set["locations"],
                                                                         query_paths, self.database_images_set["images_path"], model, version)
                del images_high, images_low, locations, features_low, vectors_low

            recall_rate = {model: {version: recall_score[model][version] / len(self.query_images_set) for version in ["pretrained", "trained"]} for model in recall_score}

            for model, rate in recall_rate.items():
                for version in ["pretrained", "trained"]:
                    print(f"{model} {version} recall rates ({', '.join([f'R@{k}' for k in self.topk_nodes.tolist()])}): {', '.join([str(score) for score in rate[version][0].tolist()])}")
            
            del recall_score, recall_rate
        torch.cuda.empty_cache()
        for model_name, save_path in self.retrieval_save_path.items():
            retrieved_database = pd.DataFrame.from_dict(self.retrieval_database[model_name])
            retrieved_database.to_pickle(save_path)

    
    def _compute_image_descriptors(self, input_images, models, calc_pos_neg=False):
        # Batches containing identical resolution images are stacked into tensors
        if isinstance(input_images, torch.Tensor):
            images = (input_images[:, :, 0, :, :, :].view(-1, input_images.shape[2:]) if calc_pos_neg else input_images[:, 0, 0, :, :, :]).to(self.device)
        # otherwise batches contain lists, which each have the query image and its positive & negative images
        else:
            images = []
            for batch_item in input_images:
                images.extend([batch_item[i].to(self.device) for i in range(len(batch_item) if calc_pos_neg else 1)])
        # Compute encoded features and global descriptors for images
        with torch.autocast('cuda', torch.float32):
            if isinstance(images, torch.Tensor):
                outputs = {model: {"pretrained": self.extrs_pretrained(model, images), "trained": self.extrs_trained(model, images)} for model in models}
            else:
                # Given different query image resolutions, compute sequentially
                pretrained_output = {model: [self.extrs_pretrained(model, image) for image in images] for model in models}
                trained_output = {model: [self.extrs_trained(model, image) for image in images] for model in models}
                outputs = {model: {"pretrained": [[img_out[0] for img_out in pretrained_output[model]], torch.cat([img_out[1] for img_out in pretrained_output[model]])],
                                   "trained": [[img_out[0] for img_out in trained_output[model]], torch.cat([img_out[1] for img_out in trained_output[model]])]} for model in models}
        encodings, descriptors = [{model: {version: data[i] for version, data in outputs[model].items()} for model in outputs} for i in range(2)]
        return encodings, descriptors


def main(configs, data_info):
    root = configs['root']
    data_name = configs['test_data']['name']

    test_data_dir_name = data_info['testsets_path']
    database_folder = data_info[data_name]['database']
    query_folder = data_info[data_name]['query']
    subset = data_info[data_name]['subset']

    data_folders = {
        "database": join(root, test_data_dir_name, data_name, subset, database_folder),
        "query": join(root, test_data_dir_name, data_name, subset, query_folder)
    }
    if "valid" in data_info[data_name]: data_folders["valid"] = join(root, test_data_dir_name, data_name, subset, data_info["valid"])
    
    vpr = VPRTester(root, configs['test_data'], data_folders, configs['vpr'], configs['train_conf'], configs["model_IO"], configs["eval_batch"])
    vpr.validation(iter_num)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../configs/test_trained_model.yaml')
    parser.add_argument("-d", "--data_info", type=str, default="../configs/testing_data.yaml")
    parser.add_argument("--test_set", type=str, help="Name of dataset to use")
    args = parser.parse_args()

    with open(args.data_info, "r") as d_locs_file:
        data_info = yaml.safe_load(d_locs_file)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    if args.test_set is not None:
        config["test_data"]["name"] = args.test_set
    config["test_data"]["name"] = dataset
    main(config, data_info)

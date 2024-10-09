import subprocess
import numpy as np
import torch.nn as nn
import yaml
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from feature.Geomatric_verification import Geometric_verification
from os.path import basename, exists, join
from os import makedirs

from dataset.Data_Control import collate_fn, load_pitts250k_data, BatchSampler

from loss.loss_distill import Loss_distill
from math import ceil
import h5py
import faiss
from feature.Global_Extractors import GlobalExtractors
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import ConcatDataset
import torch
import torch.backends.cuda as cuda
import torch.backends.cudnn as cudnn
import torch.backends.opt_einsum as opt_einsum
import random
import scipy.io
import shutil

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
cudnn.benchmark = True
cuda.matmul.allow_fp16_reduced_precision_reduction = True
cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.set_default_dtype(torch.float32)
opt_einsum.enabled = True
cudnn.enabled = True

def split_integer(x,n):
    quotient,remainder=divmod(x,n)
    slices=[quotient for _ in range(n)]
    for i in range(remainder):
        slices[i]+=1
    return slices

class VPR():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, configs, data_info, data_name):
        train_conf, self.train_conf = configs["train"], configs["train"]
        extractors_conf = configs["vpr"]["global_extractor"]
        self.batch_size = train_conf["batch_size"]
        self.lr = train_conf["lr"]
        self.lr_decay = train_conf["lr_decay"]
        self.gamma = train_conf["exponential_gamma"]
        self.thresholds = torch.tensor(configs['vpr']['threshold'])
        self.topk_nodes=torch.tensor(configs['vpr']['topk'])

        self.teacher_models = GlobalExtractors(configs["root"], extractors_conf, data_parallel=train_conf["multiGPU"])

        logs_root = join(configs["root"], configs["model_IO"]["logs_path"], data_name)
        makedirs(logs_root, exist_ok=True)
        weights_root = join(configs["root"], configs["model_IO"]["weights_path"])
        self.weight_paths, self.log_dirs = {}, {}
        log_suffix = join(data_name + ("_distill" if train_conf["loss"]["distill"] else "") + ("_vlad" if train_conf["loss"]["vlad"] else "") \
                          + ("_triplet" if train_conf["loss"]["triplet"] else ""), "{}",
                          f"{train_conf['data']['resolution']}_{train_conf['data']['qp']}_{self.lr}")

        for model_name in self.teacher_models.models:
            model_suffix = log_suffix.format(model_name)
            self.weight_paths[model_name] = join(weights_root, model_suffix)
            self.log_dirs[model_name] = join(logs_root, model_suffix)
            makedirs(self.weight_paths[model_name], exist_ok=True)
            if train_conf["resume"]:
                extractors_conf[model_name]["ckpt_path"] = self.weight_paths[model_name]

        self.log_writers = {model: SummaryWriter(log_dir=log_dir) for model, log_dir in self.log_dirs.items()}
        self.student_models = GlobalExtractors(configs["root"], extractors_conf, pipeline=train_conf["resume"], data_parallel=train_conf["multiGPU"])

        self.start_epochs = {model: self.student_models.last_epoch(model) + 1 if train_conf["resume"] else 0 for model in self.student_models.models}
        self.best_scores = {model: self.student_models.best_score(model) if train_conf["resume"] else 0 for model in self.student_models.models}
        triplet_loss_config = train_conf["triplet_loss"]
        self.nPosSample,self.nNegSample=triplet_loss_config['nPosSample'],triplet_loss_config['nNegSample']

        ### loading database / query data
        self.databases = self.load_database(data_info["database"])    # MEMORY: 10970 MiB GPU, 17.3G CPU
        self.query_data_sets = load_pitts250k_data(data_info["query"], train_conf, self.student_models.models)
        self.train_data_sets = load_pitts250k_data(data_info["train"], train_conf, self.student_models.models)
        self.valid_data_sets = load_pitts250k_data(data_info["valid"], train_conf, self.student_models.models)

        self.query_data_set=ConcatDataset(self.query_data_sets)    # MEMORY: 10970 MiB GPU, 34.0G CPU
        self.train_data_set=ConcatDataset(self.train_data_sets)
        self.valid_data_set=ConcatDataset(self.valid_data_sets)

        self.initcache = join(configs["root"],"logs",data_name,"VID" + "_" + str(train_conf["num_cluster"]) + "_"+str(train_conf["data"]["resolution"]) +"_"+str(train_conf["data"]["qp"])+".hdf5")

        self.geometric_verification = Geometric_verification()

        self.similarity_loss=Loss_distill().to(self.device)
        self.vlad_mse_loss=nn.MSELoss().to(self.device)
        self.triplet_loss = nn.TripletMarginLoss(margin=triplet_loss_config["margin"] ** 0.5, p=2, reduction="sum").to(self.device)
        
        self.sigmoid = nn.Sigmoid()

    #@pysnooper.snoop('train-VID.log', depth=3)
    def load_database(self, db_info):
        databases={}
        image_folder = db_info["image_folder"]

        # Create descriptor file handles and empty descriptor tensors
        g_desc_files = {model: h5py.File(join(image_folder, f"global_descriptor_{model}.h5"), "r") for model in self.teacher_models.models}
        num_images = len(g_desc_files[list(g_desc_files.keys())[0]][basename(image_folder)])
        assert all([num_images == len(file[basename(image_folder)]) for file in g_desc_files.values()]), \
            f"Database global descriptors for {image_folder} have different lengths (first file has length {num_images})."
        descriptors = {model: torch.empty((num_images, self.teacher_models.feature_length(model))) for model in self.teacher_models.models}
        locations = torch.empty((num_images, 2))
        names=[]
        for file_ind, (model, g_desc_file) in enumerate(g_desc_files.items()):
            for i, (name, descriptor) in enumerate(tqdm(g_desc_file[basename(image_folder)].items(), desc=f"Reading database info for {model}")):
                if file_ind == 0:
                    pitch, yaw, image_name = name.split("+")
                    image_path = join(image_folder, pitch, yaw, "raw", image_name)
                    names.append(image_path)
                    locations[i] = torch.tensor(db_info["utm"][db_info["id"].index(int(image_name.replace(".jpg", "")))])
                descriptors[model][i] = torch.tensor(descriptor.__array__())
            g_desc_file.close()

        databases['database']={'images_path':names,'descriptors':descriptors,'locations':locations}
        return databases

    def train_student(self,epoch):
        
        self.optimizers = {model: optim.Adam(self.student_models.model_parameters(model), lr=self.lr, weight_decay=self.lr_decay, foreach=True)
                           for model in self.student_models.models}

        schedulers = {model: ExponentialLR(optimizer, gamma=self.gamma) for model, optimizer in self.optimizers.items()}
        self.student_models.set_train(True)

        # All training images are used
        batch_sampler = BatchSampler([list(range(len(self.train_data_set)))], self.batch_size)
        data_loader = DataLoader(self.train_data_set, batch_sampler=batch_sampler, num_workers=self.train_conf["num_worker"], collate_fn=collate_fn, pin_memory=True)
        # total=sample_size*len(self.train_data_sets)//self.batch_size+1
        
        total = len(self.train_data_set) // self.batch_size + 1

        scaler = GradScaler()

        enabled_loss_types = set(loss_type for loss_type, enabled in self.train_conf["loss"].items() if enabled).union(["loss"])
        models_losses = {model: {loss_type: 0 for loss_type in enabled_loss_types} for model in self.student_models.models}

        for images_high, images_low, _, _ in (pbar:=tqdm(data_loader, total=total)):
            for optimizer in self.optimizers.values(): optimizer.zero_grad()
            curr_losses = {model: {loss_type: 0 for loss_type in enabled_loss_types} for model in models_losses}
            features_low, vectors_low = self._compute_image_descriptors(images_low, self.student_models, models_losses.keys(), True)
            with torch.no_grad():
                features_high, vectors_high = self._compute_image_descriptors(images_high, self.teacher_models, models_losses.keys(), True)
            for model, losses in curr_losses.items():
                with torch.autocast("cuda", torch.float32):
                    if "distill" in enabled_loss_types:
                        loss_sp = self.similarity_loss(features_low[model], features_high[model]) / 1000
                        losses["distill"] += loss_sp
                        losses["loss"] += loss_sp
                    if "vlad" in enabled_loss_types:
                        loss_vlad = self.vlad_mse_loss(vectors_low[model], vectors_high[model]) * len(images_high) * 100
                        losses["vlad"] += loss_vlad
                        losses["loss"] += loss_vlad
                    if "triplet" in enabled_loss_types:
                        loss_triplet, vector_low = 0, vectors_low[model].view(*images_high[model].shape[:2], -1)
                        for pos_neg_vecs in vector_low:
                            vladQ, vladP, vladN = torch.split(pos_neg_vecs, [1, self.nPosSample, self.nNegSample])
                            vladQ = vladQ.squeeze()
                            for vP in vladP:
                                for vN in vladN:
                                    loss_triplet += self.triplet_loss(vladQ, vP, vN)
                        loss_triplet /= (len(images_high) * self.nPosSample * self.nNegSample * 10)
                        losses["triplet"] += loss_triplet
                        losses["loss"] += loss_triplet
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(self.optimizers[model])
                clip_grad_norm_(self.student_models.model_parameters(model), 1, foreach=True)
                scaler.step(self.optimizers[model])
                schedulers[model].step()
                scaler.update()

            loss_desc = " ".join([f"{model} ep#{self.start_epochs[model] + epoch} loss: {losses['loss']:.4e} (lr: {self.optimizers[model].param_groups[0]['lr']:.4e})"
                                    for model, losses in curr_losses.items()])
            progress_desc = f"Epoch {epoch + 1}/{self.train_conf['nepoch']} " + loss_desc
            pbar.set_description(progress_desc, refresh=False)
            pbar.update(1)

            for model, losses in curr_losses.items():
                for loss_name, loss_value in losses.items():
                    models_losses[model][loss_name] += loss_value
            del images_high, images_low, curr_losses, features_low, vectors_low, features_high, vectors_high
            torch.cuda.empty_cache()
        
        for model, losses in models_losses.items():
            for loss_type, loss_value in losses.items():
                self.log_writers[model].add_scalar(f"Train/{'Loss' if loss_type == 'loss' else f'Loss_{loss_type}'}",
                                                   loss_value / len(self.train_data_set), self.start_epochs[model] + epoch)

    def vpr_examing(self,query_desc,database_desc, query_loc, database_loc):
        query_desc,database_desc, query_loc, database_loc=query_desc.to(self.device),database_desc.to(self.device), query_loc.float().to(self.device), database_loc.float().to(self.device)
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
                    if len(index_match)>0:
                        maskA = torch.zeros((self.thresholds.shape[0], self.topk_nodes.shape[0]), dtype=torch.bool)
                        maskB = torch.zeros((self.thresholds.shape[0], self.topk_nodes.shape[0]), dtype=torch.bool)
                        maskA[self.thresholds >= thre, :] = True
                        maskB[:, self.topk_nodes > torch.min(ind[0]).item()] = True
                        mask = (maskA & maskB)
                        success_num[mask] += 1
                        matched = True


            del qloc,dloc,distance
        del sim,topk,query_desc,database_desc, query_loc, database_loc
        torch.cuda.empty_cache()
        return success_num

    def validation(self,ind):
        print('start vpr examing ...')

        self.student_models.set_train(False)

        random.seed(10)
        
        # All query images are evaluated for VPR recall
        query_batch=20
        batch_sampler=BatchSampler([list(range(len(self.query_data_set)))], query_batch)
        data_loader = DataLoader(self.query_data_set, batch_sampler=batch_sampler, num_workers=self.train_conf["num_worker"], collate_fn=collate_fn, pin_memory=True)
        total = len(self.query_data_set) // query_batch + 1

        recall_score = {model: {version: torch.zeros((self.thresholds.shape[0], self.topk_nodes.shape[0])) for version in ["teacher", "student"]} for model in self.teacher_models.models}
        with torch.no_grad():
            for _, images_low, images_low_path, locations in tqdm(data_loader, total=total):
                database=self.databases['database']
                features_low, vectors_low = self._compute_image_descriptors(images_low, self.student_models, recall_score.keys())
                features_teacher_low, vectors_teacher_low = self._compute_image_descriptors(images_low, self.teacher_models, recall_score.keys())

                for model in vectors_low:
                    locations_model = locations[model][:,0,:]
                    recall_score[model]["student"] += self.vpr_examing(vectors_low[model], database["descriptors"][model], locations_model, database["locations"])
                    recall_score[model]["teacher"] += self.vpr_examing(vectors_teacher_low[model], database["descriptors"][model], locations_model, database["locations"])
                del images_low, images_low_path, locations, features_teacher_low, features_low, vectors_low, vectors_teacher_low

            recall_rate = {model: {version: recall_score[model][version] / len(self.query_data_set) for version in ["teacher", "student"]} for model in recall_score}

            for model, rate in recall_rate.items():
                for i, topk in enumerate(self.topk_nodes):
                    self.log_writers[model].add_scalars(f"Recall rate/@{int(topk)}", {version: rate[version][0,i] for version in rate}, self.start_epochs[model] + ind)
                for version in ["teacher", "student"]:
                    print(f"{model} {version} recall rates ({', '.join([f'R@{k}' for k in self.topk_nodes.tolist()])}): {', '.join([str(score) for score in recall_rate[model][version][0].tolist()])}")

            for model in self.student_models.models:
                new_state = {"epoch": self.start_epochs[model] + ind, "best_score": recall_score[model]["student"][0, 0]}
                save_path = join(self.weight_paths[model], "checkpoint.pth")
                self.student_models.save_state(model, save_path, new_state)
                if new_state["best_score"] >= self.best_scores[model]:
                    shutil.copyfile(save_path, join(self.weight_paths[model], "model_best.pth"))
                    self.best_scores[model] = new_state["best_score"]

            del recall_score, recall_rate

        print('start loss validation ...')
        # All validation images are used
        batch_sampler = BatchSampler([list(range(len(self.valid_data_set)))], self.batch_size)
        data_loader = DataLoader(self.valid_data_set, batch_sampler=batch_sampler, num_workers=self.train_conf["num_worker"], collate_fn=collate_fn, pin_memory=True)
        total = len(self.valid_data_set) // self.batch_size + 1

        with torch.no_grad():
            enabled_loss_types = set(loss_type for loss_type, enabled in self.train_conf["loss"].items() if enabled).union(["loss"])
            models_losses = {model: {loss_type: 0 for loss_type in enabled_loss_types} for model in self.student_models.models}
            for images_high, images_low, _, _ in tqdm(data_loader, total=total):
                features_low, vectors_low = self._compute_image_descriptors(images_low, self.student_models, models_losses.keys(), True)
                features_high, vectors_high = self._compute_image_descriptors(images_high, self.teacher_models, models_losses.keys(), True)
                for model, losses in models_losses.items():
                    with torch.autocast("cuda", torch.float32):
                        loss_sp = self.similarity_loss(features_low[model], features_high[model]).cpu() / 1000    
                        loss_vlad = self.vlad_mse_loss(vectors_low[model], vectors_high[model]).cpu() * len(images_high) * 100
                        loss_triplet, vector_low = 0, vectors_low[model].view(*images_high[model].shape[:2], -1)
                        for pos_neg_vecs in vector_low:
                            vladQ, vladP, vladN = torch.split(pos_neg_vecs, [1, self.nPosSample, self.nNegSample])
                            vladQ = vladQ.squeeze()
                            for vP in vladP:
                                for vN in vladN:
                                    loss_triplet += self.triplet_loss(vladQ, vP, vN)
                        loss_triplet /= (self.nPosSample * self.nNegSample * 10)
                        loss_triplet = loss_triplet.cpu()

                        if "distill" in enabled_loss_types:
                            losses["distill"] += loss_sp
                        if "vlad" in enabled_loss_types:
                            losses["vlad"] += loss_vlad
                        if "triplet" in enabled_loss_types:
                            losses["triplet"] += loss_triplet
                        losses["loss"] = loss_sp + loss_vlad + loss_triplet

                del images_high, images_low, features_low, features_high, vectors_low, vectors_high
                torch.cuda.empty_cache()
            
        for model, losses in models_losses.items():
            for loss_type, loss_value in losses.items():
                self.log_writers[model].add_scalar(f"Valid/{'Loss' if loss_type == 'loss' else f'Loss_{loss_type}'}",
                                                   loss_value / len(self.valid_data_set), self.start_epochs[model] + ind)
    
    def _compute_image_descriptors(self, input_images, global_extractors, models, calc_pos_neg=False):
        # Remove batch dimension from images
        images = {model: (model_input.view(-1, *model_input.shape[2:]) if calc_pos_neg else
                          model_input[:, 0, :, :, :]).to(self.device) for model, model_input in input_images.items()}
        # Compute encoded features and global descriptors for images
        with torch.autocast("cuda", torch.float32):
            outputs = {model: global_extractors(model, images[model]) for model in models}
        encodings, descriptors = [{model: data[i] for model, data in outputs.items()} for i in range(2)]
        return encodings, descriptors


def extract_frames(config,image_path,video_path,PITCH,YAW):
    if not exists(image_path):
        makedirs(image_path)
        images=join(image_path,f'{PITCH}_{YAW}_%05d.png')
        cmd = [
        'ffmpeg',
        '-i', video_path,
        '-r', str(config['rate']),
        images]
        subprocess.call(cmd)

def main(configs):
    root = configs['root']
    with open(join(root,'configs','pitts250_train_data.yaml'), 'r') as f:
        data_split_config = yaml.safe_load(f)

    name=data_split_config['name']
    train_folder=data_split_config['train'][0]
    valid_folder=data_split_config['valid'][0]
    database_folder=data_split_config['database'][0]
    query_folder=data_split_config['query'][0]

    gt_path=join(root,'logs',name,'pittsburgh_database_10586_utm.mat')
    database_gt = scipy.io.loadmat(gt_path)['Cdb'].T
    database_id=[i for i,_ in enumerate(database_gt)]

    gt_path=join(root,'logs',name,'pittsburgh_query_1000_utm.mat')
    query_gt = scipy.io.loadmat(gt_path)['Cq'].T
    id_path=join(root,'logs',name,'pittsburgh_queryID_1000.mat')
    query_id = scipy.io.loadmat(id_path)['query_id'][0]
    query_id=[int(i)-1 for i in query_id]

    data={}
    data['train']={'image_folder':join(root,'logs',name,train_folder),'utm':database_gt,'id':database_id}
    data['valid']={'image_folder':join(root,'logs',name,valid_folder),'utm':database_gt,'id':database_id}
    data['database']={'image_folder':join(root,'logs',name,database_folder),'utm':database_gt,'id':database_id}
    data['query']={'image_folder':join(root,'logs',name,query_folder),'utm':query_gt,'id':query_id}

    vpr=VPR(configs,data,name)

    for epoch in range(0, configs["train"]["nepoch"]):
        vpr.train_student(epoch)
        vpr.validation(epoch)
    for log_writer in vpr.log_writers.values(): log_writer.close()
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="../configs/trainer.yaml")
    parser.add_argument("-l", "--loss-num", type=int, help="The loss combination to use (index [1-7] for the permutations of the 3 losses)")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # Allows for choosing 1 combination of the 3 available losses through specifying an integer
    if args.loss_num is not None:
        loss_combs = [(True, True, True), (True, True, False), (True, False, True), (True, False, False), (False, True, True), (False, True, False), (False, False, True)]
        for enabled, loss_type in zip(loss_combs[args.loss_num - 1], config["train"]["loss"]):
            config["train"]["loss"][loss_type] = enabled
    main(config)
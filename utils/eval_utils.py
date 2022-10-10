import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_SB, CLAM_MB
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import random
from sklearn.neighbors import NearestNeighbors
import openslide
import math

import glob
from PIL import Image
import ast

from datasets.dataset_h5 import Whole_Slide_Bag_FP
from models.resnet_custom import resnet50_baseline
from datasets.dataset_generic import Generic_MIL_Dataset


def initiate_model(args, ckpt_path):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    print_network(model)
    
    if args.cpu_only:
        ckpt = torch.load(ckpt_path,map_location=torch.device('cpu'))
    else:
         ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)
    model.relocate()
    model.eval()
    return model


def extract_features(args,loader,feature_extraction_model,use_cpu):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.cpu_only:
        device=torch.device("cpu")
    for count, (batch,coords) in enumerate(loader):
        batch = batch.to(device, non_blocking=True)
        with torch.no_grad():
            features = feature_extraction_model(batch)
        if use_cpu:
            features=features.cpu()
        if count==0:
            all_features=features
        else:
            all_features=torch.cat((all_features,features))
    if use_cpu:
        all_features=all_features.to(device)
    return all_features


def generate_sample_idxs(idxs_length,previous_samples,sampling_weights,samples_per_epoch,num_random):
    nonrandom_idxs=list(np.random.choice(range(idxs_length),p=sampling_weights,size=int(samples_per_epoch-num_random),replace=False))
    previous_samples=previous_samples+nonrandom_idxs
    available_idxs=list(set(range(idxs_length))-set(previous_samples))
    random_idxs=list(np.random.choice(available_idxs, size=num_random,replace=False))
    sample_idxs=random_idxs+nonrandom_idxs
    return sample_idxs


def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    print("model on device:",next(model.parameters()).device)
    print('Init Loaders')
    
    if args.eval_features:
        patient_results, test_error, auc, df, _ = summary_eval_features(model,dataset,args)

    elif args.sampling:
        assert 0<=args.sampling_random<=1,"sampling_random needs to be between 0 and 1"
        dataset.load_from_h5(True)
        loader = get_simple_loader(dataset)
        patient_results, test_error, auc, df, _ = summary_sampling(model, loader, args)
    else:
        loader = get_simple_loader(dataset)
        patient_results, test_error, auc, df, _ = summary(model, loader, args)
        
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, df


def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(data)
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger


def summary_eval_features(model,dataset,args):
    model.eval()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.cpu_only:
        device=torch.device("cpu")
    feature_extraction_model=resnet50_baseline(pretrained=True,dataset=args.pretraining_dataset)
    feature_extraction_model = feature_extraction_model.to(device)
    
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)

    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(dataset), args.n_classes))

    label_dict=ast.literal_eval(args.label_dict)
    all_labels = pd.read_csv(args.csv_path)['label']
    all_labels=[label_dict[key] for key in all_labels]
    all_labels_tensor=torch.Tensor(all_labels)
    all_preds = np.zeros(len(dataset))
    patient_results = {}
    
    num_random=int(args.samples_per_epoch*args.sampling_random)

    ## Collecting Y_hats and labels to view performance across sampling epochs
    Y_hats=[]
    labels=[]
    Y_probs=[]
    all_logits=[]
    total = len(dataset)
       
    kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}    
    for bag_candidate_idx in range(total):
        label=all_labels[bag_candidate_idx]
        label_tensor=all_labels_tensor[bag_candidate_idx]
        if isinstance(dataset[bag_candidate_idx],np.int64):
            slide_id=str(dataset[bag_candidate_idx])
        else:
            slide_id = dataset[bag_candidate_idx].split(args.slide_ext)[0]
        bag_name = slide_id+'.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        
        wsi = openslide.open_slide(slide_file_path)
        sampled_data = Whole_Slide_Bag_FP(file_path=h5_file_path, wsi=wsi, pretrained=True,
                            custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size)

        all_coords=sampled_data.coords(len(sampled_data))
        
        X = np.array(all_coords)
        nbrs = NearestNeighbors(n_neighbors=args.sampling_neighbors, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        samples_per_epoch=args.samples_per_epoch
        if args.samples_per_epoch>len(all_coords):
            samples_per_epoch=len(all_coords)
            print("full slide used")
            sample_idxs=range(len(all_coords))
            
        else:
            sample_idxs=list(np.random.choice(range(0,len(all_coords)), size=samples_per_epoch,replace=False))
        
        all_sample_idxs=sample_idxs
        sampled_data.update_sample(sample_idxs)
        loader = DataLoader(dataset=sampled_data, batch_size=args.batch_size, **kwargs, collate_fn=collate_features)
        
        if len(sample_idxs)>20000:
            all_features=extract_features(args,loader,feature_extraction_model,use_cpu=True)
        else:
            all_features=extract_features(args,loader,feature_extraction_model,use_cpu=False)
            
        all_previous_features=all_features
        all_sample_idxs=sample_idxs
        
        if args.cpu_only:
            device=torch.device("cpu")
            all_features.to(device)
        with torch.no_grad():
            logits, Y_prob, Y_hat, raw_attention, results_dict = model(all_features)
        attention_scores=torch.nn.functional.softmax(raw_attention,dim=1)[0].cpu()
        attn_scores_list=raw_attention[0].cpu().tolist()


        if not args.use_all_samples:
            if args.samples_per_epoch<=args.retain_best_samples:
                best_sample_idxs=sample_idxs
                best_attn_scores=attn_scores_list
            else:
                attn_idxs=[idx.item() for idx in np.argsort(attn_scores_list)][::-1]
                best_sample_idxs=[sample_idxs[attn_idx] for attn_idx in attn_idxs][:args.retain_best_samples]
                best_attn_scores=[attn_scores_list[attn_idx] for attn_idx in attn_idxs][:args.retain_best_samples]

        all_attentions=attention_scores
        Y_hats.append(Y_hat)
        Y_probs.append(Y_prob)
        all_logits.append(logits)

        sampling_random=args.sampling_random

        ## Subsequent epochs
        neighbors=args.sampling_neighbors
        sampling_weights=np.zeros(len(all_coords))
    
        for epoch_count in range(args.sampling_epochs-1):
            sampling_random=max(sampling_random-args.sampling_random_delta,0)
            num_random=int(samples_per_epoch*sampling_random)

            attention_scores=attention_scores/max(attention_scores)
            all_attentions=all_attentions/max(all_attentions)

            if args.sampling_average:
                for i in range(len(sample_idxs)):
                    ##Loop through neighbors 
                    for index in indices[sample_idxs[i]][:neighbors]:
                        ##Update the newly found weights
                        if sampling_weights[index]>0:
                            sampling_weights[index]=(sampling_weights[index]+pow(attention_scores[i],0.15))/2
                        else:
                            sampling_weights[index]=pow(attention_scores[i],0.15)

            else:
                for i in range(len(sample_idxs)):
                    ##Loop through neighbors of the previously sampled index
                    for index in indices[sample_idxs[i]][:neighbors]:
                        #Update the newly found weights
                        sampling_weights[index]=max(sampling_weights[index],pow(attention_scores[i],0.15))
            
            ##remove previous sample weight to remove repeats
            for sample_idx in all_sample_idxs:
                sampling_weights[sample_idx]=0
            sampling_weights=sampling_weights/max(sampling_weights)
            sampling_weights=sampling_weights/sum(sampling_weights)
            sample_idxs=generate_sample_idxs(len(all_coords),all_sample_idxs,sampling_weights,samples_per_epoch,num_random)
            all_sample_idxs=all_sample_idxs+sample_idxs   
            
            if args.use_all_samples:
                if epoch_count==args.sampling_epochs-2:
                    for sample_idx in all_sample_idxs:
                        sampling_weights[sample_idx]=0
                    sampling_weights=sampling_weights/max(sampling_weights)
                    sampling_weights=sampling_weights/sum(sampling_weights)
                    final_sample_idxs=list(np.random.choice(range(0,len(all_coords)),p=sampling_weights,size=int(args.final_sample_size),replace=False))
                    sample_idxs=list(set(sample_idxs+final_sample_idxs))
                    all_sample_idxs=list(set(all_sample_idxs+sample_idxs))
                    
            else:
                if epoch_count==args.sampling_epochs-2:
                    if args.final_sample_size>len(all_coords):
                        sample_idxs=list(np.random.choice(range(0,len(all_coords)),p=sampling_weights,size=len(coords),replace=False))
                        print("final sample using all coords")
                    else:
                        for sample_idx in all_sample_idxs:
                            sampling_weights[sample_idx]=0
                        sampling_weights=sampling_weights/max(sampling_weights)
                        sampling_weights=sampling_weights/sum(sampling_weights)
                        
                        sample_idxs=list(np.random.choice(range(0,len(all_coords)),p=sampling_weights,size=int(args.final_sample_size-len(best_sample_idxs)),replace=False))
                        sample_idxs=list(set(sample_idxs+best_sample_idxs))
            
            sampled_data.update_sample(sample_idxs)
            loader = DataLoader(dataset=sampled_data, batch_size=args.batch_size, **kwargs, collate_fn=collate_features)

            all_features=extract_features(args,loader,feature_extraction_model,use_cpu=False)
            all_previous_features=torch.cat((all_previous_features,all_features))
                
            if args.use_all_samples:
                if epoch_count==args.sampling_epochs-2: 
                    all_features=all_previous_features

            all_features=all_features.to(device)
            with torch.no_grad():
                logits, Y_prob, Y_hat, raw_attention, results_dict = model(all_features)
            attention_scores=torch.nn.functional.softmax(raw_attention,dim=1)[0].cpu()
            all_attention=attention_scores
            attention_scores=attention_scores[-samples_per_epoch:]
        
            attn_scores_list=raw_attention[0].cpu().tolist()
            
            if not args.use_all_samples:
                attn_scores_combined=attn_scores_list+best_attn_scores
                idxs_combined=sample_idxs+best_sample_idxs
        
                if len(idxs_combined)<=args.retain_best_samples:
                    best_sample_idxs=idxs_combined
                    best_attn_scores=attn_scores_combined
                else:
                    attn_idxs=[idx.item() for idx in np.argsort(attn_scores_combined)][::-1]
                    best_sample_idxs=[idxs_combined[attn_idx] for attn_idx in attn_idxs][:args.retain_best_samples]
                    best_attn_scores=[attn_scores_combined[attn_idx] for attn_idx in attn_idxs][:args.retain_best_samples]
        
            Y_hats.append(Y_hat)
            labels.append(label)
            Y_probs.append(Y_prob)
            all_logits.append(logits)

            neighbors=neighbors-args.sampling_neighbors_delta
        print("final sample size:",len(all_features))
        acc_logger.log(Y_hat, label)

        probs = Y_prob.cpu().numpy()

        all_probs[bag_candidate_idx] = probs
        all_preds[bag_candidate_idx] = Y_hat.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': float(label)}})
        error = calculate_error(Y_hat, label_tensor)
        test_error += error

    all_errors=[]
    for i in range(args.sampling_epochs):
        all_errors.append(round(calculate_error(torch.Tensor(Y_hats[i::args.sampling_epochs]),torch.Tensor(all_labels)),3))

    all_aucs=[]
    for i in range(args.sampling_epochs):
        if len(np.unique(all_labels)) == 2:
            auc_score = roc_auc_score(all_labels,[yprob.tolist()[0][1] for yprob in Y_probs[i::args.sampling_epochs]])
        else:
            assert 1==2,"AUC scoring by epoch not implemented for multi-class classification yet"
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
        
        all_aucs.append(round(auc_score,3))

    del dataset
    
    test_error /= total
    
    aucs = []
    if len(np.unique(all_labels)) == 2:
        auc_score = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        assert  1==2,"AUC scoring by epoch not implemented for multi-class classification yet"

    results_dict = {'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    print("all errors: ",all_errors)
    print("all aucs: ",all_aucs)
    return patient_results, test_error, auc_score, df, acc_logger



def summary_sampling(model, loader, args):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)

    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))
    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    
    num_random=int(args.samples_per_epoch*args.sampling_random)
    
    ## Collecting Y_hats and labels to view performance across sampling epochs
    Y_hats=[]
    labels=[]
    Y_probs=[]
    all_logits=[]
    
    if args.sampling_type=='textural':
        if args.texture_model=='levit_128s':
            texture_dataset =  Generic_MIL_Dataset(csv_path = 'dataset_csv/set_all.csv',
                        data_dir= os.path.join(args.data_root_dir, 'levit_128s'),
                        shuffle = False,
                        print_info = True,
                        label_dict = {'high_grade':0,'low_grade':1,'clear_cell':1,'endometrioid':1,'mucinous':1},
                        patient_strat= False,
                        ignore=[])
            slide_id_list = list(pd.read_csv('dataset_csv/set_all.csv')['slide_id'])


    for batch_idx, (data, label,coords,slide_id) in enumerate(loader):
        print("Processing WSI number ", batch_idx)
        coords=torch.tensor(coords)
        if args.sampling_type=='spatial':
            X = np.array(coords)
        elif args.sampling_type=='textural':
            if args.texture_model=='resnet50':
                X = np.array(data)
            elif args.texture_model=='levit_128s':
                print(slide_id[0][0])
                texture_index=slide_id_list.index(slide_id[0][0])
                levit_features=texture_dataset[texture_index][0]
                assert len(levit_features)==len(data),"wrong features accessed, code must be broken"
                X = np.array(levit_features)

            else:
                assert 1==2,'incorrect texture model chosen'
        data, label, coords = data.to(device), label.to(device), coords.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        
        samples_per_epoch=args.samples_per_epoch
        if args.samples_per_epoch>len(coords):
            samples_per_epoch=len(coords)
            print("full slide used")
                
        if args.initial_grid_sample:
            ## First epoch (random grid sampling)
            x_coords=[x.item() for x,y in coords]
            y_coords=[y.item() for x,y in coords]
            min_x=min(x_coords)
            max_x=max(x_coords)
            min_y=min(y_coords)
            max_y=max(y_coords)

            num_of_splits=int(math.sqrt(samples_per_epoch))
            x_borders=np.linspace(min_x,max_x+0.00001,num_of_splits+1)
            y_borders=np.linspace(min_y,max_y+0.00001,num_of_splits+1)

        
            sample_idxs=[]
            coords_splits=[[] for _ in range((num_of_splits+1)*(num_of_splits+1))]
            for coord_idx, (x,y) in enumerate(coords):
                x_border_idx=np.where(x_borders==max(x_borders[x_borders<=x.item()]))[0][0]
                y_border_idx=np.where(y_borders==max(y_borders[y_borders<=y.item()]))[0][0]
                coords_splits[(num_of_splits+1)*x_border_idx+y_border_idx].append(coord_idx)
            for coords_in_split in coords_splits:
                if len(coords_in_split)>0:
                    sample_idxs=sample_idxs+list(np.random.choice(coords_in_split, size=1,replace=False))

            if len(sample_idxs)<samples_per_epoch:
                sample_idxs=sample_idxs+list(np.random.choice(range(0,len(coords)), size=samples_per_epoch-len(sample_idxs),replace=False))

        else:
            ## First epoch (fully random sampling)
            sample_idxs=list(np.random.choice(range(0,len(coords)), size=samples_per_epoch,replace=False))
        
        all_sample_idxs=sample_idxs
        data_sample=data[sample_idxs].to(device)
        with torch.no_grad():
            logits, Y_prob, Y_hat, raw_attention, results_dict = model(data_sample)

        attention_scores=torch.nn.functional.softmax(raw_attention,dim=1)[0].cpu()
        attn_scores_list=raw_attention[0].cpu().tolist()

        if not args.use_all_samples:
            if args.samples_per_epoch<=args.retain_best_samples:
                best_sample_idxs=sample_idxs
                best_attn_scores=attn_scores_list
            else:
                attn_idxs=[idx.item() for idx in np.argsort(attn_scores_list)][::-1]
                best_sample_idxs=[sample_idxs[attn_idx] for attn_idx in attn_idxs][:args.retain_best_samples]
                best_attn_scores=[attn_scores_list[attn_idx] for attn_idx in attn_idxs][:args.retain_best_samples]
                   

        all_attentions=attention_scores
        Y_hats.append(Y_hat)
        labels.append(label)
        Y_probs.append(Y_prob)
        all_logits.append(logits)
        
        ## Find nearest neighbors of each patch to prepare for spatial resampling
        nbrs = NearestNeighbors(n_neighbors=args.sampling_neighbors, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X[sample_idxs])

        sampling_random=args.sampling_random

        ## Subsequent epochs
        neighbors=args.sampling_neighbors
        sampling_weights=np.zeros(len(coords))
        for epoch_count in range(args.sampling_epochs-1):
            sampling_random=max(sampling_random-args.sampling_random_delta,0)
            num_random=int(samples_per_epoch*sampling_random)
            attention_scores=attention_scores/max(attention_scores)
            all_attentions=all_attentions/max(all_attentions)
            
            ## make this a function for update_sample_weights
            if args.sampling_average:
                for i in range(len(indices)):
                    ##Loop through neighbors of the previously sampled index
                    for index in indices[i][:neighbors]:
                        ##Update the newly found weights
                        if sampling_weights[index]>0:
                            sampling_weights[index]=(sampling_weights[index]+pow(attention_scores[i],0.15))/2
                        else:
                            sampling_weights[index]=pow(attention_scores[i],0.15)
            else:    
                for i in range(len(indices)):              
                    ##Loop through neighbors of the previously sampled index
                    for index in indices[i][:neighbors]:
                        ##Update the newly found weights
                        sampling_weights[index]=max(sampling_weights[index],pow(attention_scores[i],0.15))
            
            
            ##remove previous sample weight to reduce repeats
            for sample_idx in all_sample_idxs:
                sampling_weights[sample_idx]=0
            sampling_weights=sampling_weights/max(sampling_weights)
            sampling_weights=sampling_weights/sum(sampling_weights)
            
            sample_idxs=generate_sample_idxs(len(coords),all_sample_idxs,sampling_weights,samples_per_epoch,num_random)
            all_sample_idxs=all_sample_idxs+sample_idxs
            distances, indices = nbrs.kneighbors(X[sample_idxs])

            if args.use_all_samples:
                if epoch_count==args.sampling_epochs-2:
                    for sample_idx in all_sample_idxs:
                        sampling_weights[sample_idx]=0
                    sampling_weights=sampling_weights/max(sampling_weights)
                    sampling_weights=sampling_weights/sum(sampling_weights)
                    sample_idxs=list(np.random.choice(range(0,len(coords)),p=sampling_weights,size=int(args.final_sample_size),replace=False))
                    all_sample_idxs=all_sample_idxs+sample_idxs
                    data_sample=data[all_sample_idxs].to(device)
                else:
                    data_sample=data[sample_idxs].to(device)
            else:
                if epoch_count==args.sampling_epochs-2:
                    if args.final_sample_size>len(coords):
                        sample_idxs=list(np.random.choice(range(0,len(coords)),p=sampling_weights,size=len(coords),replace=False))
                        print("final sample using all coords")
                    else:    
                        for sample_idx in all_sample_idxs:
                            sampling_weights[sample_idx]=0
                        sampling_weights=sampling_weights/max(sampling_weights)
                        sampling_weights=sampling_weights/sum(sampling_weights)
                        sample_idxs=list(np.random.choice(range(0,len(coords)),p=sampling_weights,size=int(args.final_sample_size-len(best_sample_idxs)),replace=False))
                        sample_idxs=sample_idxs+best_sample_idxs
                    data_sample=data[sample_idxs].to(device)
                else:
                    data_sample=data[sample_idxs].to(device)
            
            with torch.no_grad():
                logits, Y_prob, Y_hat, raw_attention, results_dict = model(data_sample)
            attention_scores=torch.nn.functional.softmax(raw_attention,dim=1)[0].cpu()
            all_attention=attention_scores
            attention_scores=attention_scores[-samples_per_epoch:]
            attn_scores_list=raw_attention[0].cpu().tolist()

                        
            if not args.use_all_samples:
                attn_scores_combined=attn_scores_list+best_attn_scores
                idxs_combined=sample_idxs+best_sample_idxs


                if len(idxs_combined)<=args.retain_best_samples:
                    best_sample_idxs=idxs_combined
                    best_attn_scores=attn_scores_combined
                else:
                    attn_idxs=[idx.item() for idx in np.argsort(attn_scores_combined)][::-1]
                    best_sample_idxs=[idxs_combined[attn_idx] for attn_idx in attn_idxs][:args.retain_best_samples]
                    best_attn_scores=[attn_scores_combined[attn_idx] for attn_idx in attn_idxs][:args.retain_best_samples]

            Y_hats.append(Y_hat)
            labels.append(label)
            Y_probs.append(Y_prob)
            all_logits.append(logits)
            
            neighbors=neighbors-args.sampling_neighbors_delta

            if args.plot_sampling:          
                if epoch_count == args.sampling_epochs-2:
                    if args.use_all_samples:
                        sample_coords=coords[all_sample_idxs]
                    else:
                        sample_coords=coords[sample_idxs]
                        
                    print("Plotting epoch",epoch_count+1,"for slide",slide_id)
                    thumbnail_size=1000
                    slide = openslide.open_slide("../mount_point/"+slide_id+".svs")
                    img = slide.get_thumbnail((thumbnail_size, thumbnail_size))
                    plt.figure()
                    plt.imshow(img)
                    x_values,y_values=[(x-128)*(thumbnail_size/max(slide.dimensions)) for x,y in sample_coords.tolist()],[(y-128)*(thumbnail_size/max(slide.dimensions)) for x,y in sample_coords.tolist()]
                    plt.scatter(x_values,y_values,s=6)
                    plt.savefig('../mount_outputs/sampling_maps/{}_epoch{}.png'.format(slide_id,epoch_count+1), dpi=300)
                    plt.close()

            if args.plot_sampling_gif:
                if args.use_all_samples:
                    sample_coords=coords[all_sample_idxs]
                else:
                    sample_coords=coords[sample_idxs]
                thumbnail_size=1000
                slide = openslide.open_slide("../mount_point/"+slide_id+".svs")
                img = slide.get_thumbnail((thumbnail_size, thumbnail_size))
                plt.figure()
                plt.imshow(img)
                x_values,y_values=[(x-128)*(thumbnail_size/max(slide.dimensions)) for x,y in sample_coords.tolist()],[(y-128)*(thumbnail_size/max(slide.dimensions)) for x,y in sample_coords.tolist()]
                plt.scatter(x_values,y_values,s=6)
                plt.savefig('../mount_outputs/sampling_maps/{}_epoch{}.png'.format(slide_id,epoch_count+1), dpi=300)
                plt.close()

                if epoch_count == args.sampling_epochs-2:
                    fp_in = "../mount_outputs/sampling_maps/{}_epoch*.png".format(slide_id)
                    fp_out = "../mount_outputs/sampling_maps/{}.gif".format(slide_id)
                    imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
                    img = next(imgs)  # extract first image from iterator
                    img.save(fp=fp_out, format='GIF', append_images=imgs,
                                     save_all=True, duration=200, loop=1)

        acc_logger.log(Y_hat, label)
                 
        probs = Y_prob.cpu().numpy()
        
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
                                                         
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    all_errors=[]
    for i in range(args.sampling_epochs):
        all_errors.append(round(calculate_error(torch.Tensor(Y_hats[i::args.sampling_epochs]),torch.Tensor(labels[i::args.sampling_epochs])),3))
    
    all_aucs=[]
    for i in range(args.sampling_epochs):
        if len(np.unique(all_labels)) == 2:
            auc_score = roc_auc_score(all_labels,[yprob.tolist()[0][1] for yprob in Y_probs[i::args.sampling_epochs]])
        else:
            assert 1==2,"AUC scoring by epoch not implemented for multi-class classification yet"
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
                    if args.micro_average:
                        binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                        fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                        auc_score = auc(fpr, tpr)
                    else:
                        auc_score = np.nanmean(np.array(aucs))
        all_aucs.append(round(auc_score,3))

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 2:
        auc_score = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
        for class_idx in range(args.n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(auc(fpr, tpr))
            else:
                aucs.append(float('nan'))
        if args.micro_average:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
            auc_score = auc(fpr, tpr)
        else:
            auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    print("all errors: ",all_errors)
    print("all aucs: ",all_aucs)
    return patient_results, test_error, auc_score, df, acc_logger

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

import glob
from PIL import Image

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


def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
 
    print('Init Loaders')
    
    if args.sampling:
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
    for batch_idx, (data, label,coords,slide_id) in enumerate(loader):
        print("Processing WSI number ", batch_idx)
        coords=torch.tensor(coords)
        X = np.array(coords)
        data, label, coords = data.to(device), label.to(device), coords.to(device)
        slide_id = slide_ids.iloc[batch_idx]
            
        ## First epoch (fully random sampling)
        sample_idxs=random.sample(range(0,len(coords)), k=args.samples_per_epoch)
        all_sample_idxs=sample_idxs
        data_sample=data[sample_idxs].to(device)
        with torch.no_grad():
            logits, Y_prob, Y_hat, raw_attention, results_dict = model(data_sample)

        attention_scores=torch.nn.functional.softmax(raw_attention,dim=1)[0].cpu()
        all_attentions=attention_scores
        Y_hats.append(Y_hat)
        labels.append(label)
        Y_probs.append(Y_prob)
        all_logits.append(logits)
        
        ## Find nearest neighbors of each patch to prepare for spatial resampling
        nbrs = NearestNeighbors(n_neighbors=args.sampling_neighbors, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X) ##can use distances, indicies to try different sampling approaches     

        sampling_random=args.sampling_random

        ## Subsequent epochs
        neighbors=args.sampling_neighbors
        sampling_weights=np.zeros(len(coords))
        for epoch_count in range(args.sampling_epochs-1):
            sampling_random=max(sampling_random-args.sampling_random_delta,0)
            num_random=int(args.samples_per_epoch*sampling_random)
 
            ## Previously used random.choices which has replacement, sample does not
            random_idxs=random.sample(range(0,len(coords)), k=num_random)
            
            attention_scores=attention_scores/max(attention_scores)
            all_attentions=all_attentions/max(all_attentions)
            
            for i in range(len(sample_idxs)):              
                ##Loop through neighbors of the previously sampled index
                for index in indices[sample_idxs[i]][:neighbors]:
                    ##Update the newly found weights
                    sampling_weights[index]=max(sampling_weights[index],pow(attention_scores[i],0.15))
            
            sampling_weights=sampling_weights/max(sampling_weights)
            nonrandom_idxs=random.choices(range(0,len(coords)),weights=sampling_weights,k=int(args.samples_per_epoch-num_random))        
            sample_idxs=random_idxs+nonrandom_idxs
            all_sample_idxs=all_sample_idxs+sample_idxs
            
            if args.use_all_samples:
                if epoch_count==args.sampling_epochs-2:
                    data_sample=data[all_sample_idxs].to(device)
                else:
                    data_sample=data[sample_idxs].to(device)
            else:
                if epoch_count==args.sampling_epochs-2:
                    sample_idxs=random.choices(range(0,len(coords)),weights=sampling_weights,k=int(args.final_sample_size))
                    data_sample=data[sample_idxs].to(device)
                else:
                    data_sample=data[sample_idxs].to(device)
            
            with torch.no_grad():
                logits, Y_prob, Y_hat, raw_attention, results_dict = model(data_sample)
            attention_scores=torch.nn.functional.softmax(raw_attention,dim=1)[0].cpu()
            all_attention=attention_scores
            attention_scores=attention_scores[-args.samples_per_epoch:]
            
            Y_hats.append(Y_hat)
            labels.append(label)
            Y_probs.append(Y_prob)
            all_logits.append(logits)
            
            neighbors=neighbors-args.sampling_neighbors_delta

            if args.plot_sampling:          
                if epoch_count == args.sampling_epochs-2:
                
                    #print(all_sample_idxs)
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

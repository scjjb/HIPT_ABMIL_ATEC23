import numpy as np
import math
import openslide
import matplotlib.pyplot as plt
import glob
from PIL import Image
from matplotlib import colors

def generate_sample_idxs(idxs_length,previous_samples,sampling_weights,samples_per_iteration,num_random,grid=False,coords=None):
    if grid:
        assert len(coords)>0
        x_coords=[x.item() for x,y in coords]
        y_coords=[y.item() for x,y in coords]
        min_x=min(x_coords)
        max_x=max(x_coords)
        min_y=min(y_coords)
        max_y=max(y_coords)
        
        num_of_splits=int(math.sqrt(samples_per_iteration))
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
        if len(sample_idxs)<samples_per_iteration:
            sample_idxs=sample_idxs+list(np.random.choice(range(0,len(coords)), size=samples_per_iteration-len(sample_idxs),replace=False))

    else:
        available_idxs=set(range(idxs_length))
        nonrandom_idxs=[]
        random_idxs=[]
        if int(samples_per_iteration-num_random)>0:
            nonrandom_idxs=list(np.random.choice(range(idxs_length),p=sampling_weights,size=int(samples_per_iteration-num_random),replace=False))
            previous_samples=previous_samples+nonrandom_idxs
            available_idxs=available_idxs-set(previous_samples)
        if num_random>0:
            random_idxs=list(np.random.choice(list(available_idxs), size=num_random,replace=False))
        sample_idxs=random_idxs+nonrandom_idxs
    return sample_idxs


def generate_features_array(args, data, coords, slide_id, slide_id_list, texture_dataset):
    if args.sampling_type=='spatial':
        X = np.array(coords)
    elif args.sampling_type=='textural':
        assert args.texture_model in ['resnet50','levit_128s'], 'incorrect texture model chosen'
        if args.texture_model=='resnet50':
            X = np.array(data)
        else:
            texture_index=slide_id_list.index(slide_id[0][0])
            levit_features=texture_dataset[texture_index][0]
            assert len(levit_features)==len(data),"features length mismatch"
            X = np.array(levit_features)
    return X


def update_sampling_weights(sampling_weights, attention_scores, all_sample_idxs, indices, neighbors, power=0.15, normalise = True, sampling_update = 'max', repeats_allowed = False):
    """
    Updates the sampling weights of all patches by looping through the most recent sample and adjusting all neighbors weights
    By default the weight of a patch is the maximum of its previous weight and the newly assigned weight, though sampling_average changes this to an average
    power is a hyperparameter controlling how attention scores are smoothed as typically very close to 0 or 1
    if repeated_allowed = False then weights for previous samples are set to 0
    """
    assert sampling_update in ['max','average','none']
    if sampling_update=='average':
        for i in range(len(indices)):
            for index in indices[i][:neighbors]:
                ## the default value is 0.0001
                if sampling_weights[index]>0.0001:
                    sampling_weights[index]=(sampling_weights[index]+pow(attention_scores[i],power))/2
                else:
                    sampling_weights[index]=pow(attention_scores[i],power)
    elif sampling_update=='max':
        for i in range(len(indices)):
            #print("indices:",len(indices))
            for index in indices[i][:neighbors]:
                sampling_weights[index]=max(sampling_weights[index],pow(attention_scores[i],power))

    if not repeats_allowed:
        for sample_idx in all_sample_idxs:
            sampling_weights[sample_idx]=0

    if normalise:
        #sampling_weights=sampling_weights/max(sampling_weights)
        sampling_weights=sampling_weights/sum(sampling_weights)

    return sampling_weights


def plot_sampling(slide_id,sample_coords,args,thumbnail_size=1000):
    print("Plotting slide {} with {} samples".format(slide_id,len(sample_coords)))
    slide = openslide.open_slide(args.data_slide_dir+"/"+slide_id+".svs")
    img = slide.get_thumbnail((thumbnail_size,thumbnail_size))
    plt.figure()
    plt.imshow(img)
    x_values, y_values = sample_coords.T
    x_values=(x_values+128)*(thumbnail_size/max(slide.dimensions))
    y_values=(y_values+128)*(thumbnail_size/max(slide.dimensions))
    x_values=x_values.cpu()
    y_values=y_values.cpu()
    plt.scatter(x_values,y_values,s=6)
    plt.savefig('../mount_outputs/sampling_maps/{}.png'.format(slide_id), dpi=300)
    plt.close()
    
def plot_sampling_gif(slide_id,sample_coords,args,iteration,slide=None,final_iteration=False,thumbnail_size=1000):
    if slide==None:
        slide = openslide.open_slide(args.data_slide_dir+"/"+slide_id+".svs")
    
    img = slide.get_thumbnail((thumbnail_size,thumbnail_size))
    plt.figure()
    plt.imshow(img)
    x_values, y_values = sample_coords.T
    x_values=(x_values+128)*(thumbnail_size/max(slide.dimensions))
    y_values=(y_values+128)*(thumbnail_size/max(slide.dimensions))
    x_values=x_values.cpu()
    y_values=y_values.cpu()
    plt.scatter(x_values,y_values,s=6)
    plt.savefig('../mount_outputs/sampling_maps/{}_iter{}.png'.format(slide_id,iteration), dpi=300)
    plt.close()
    
    if final_iteration:
        print("Plotting gif for slide {} over {} iterations".format(slide_id,iteration+1))
        fp_in = "../mount_outputs/sampling_maps/{}_iter*.png".format(slide_id)
        fp_out = "../mount_outputs/sampling_maps/{}.gif".format(slide_id)
        imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
        img = next(imgs)  # extract first image from iterator
        img.save(fp=fp_out, format='GIF', append_images=imgs,save_all=True, duration=200, loop=1)

    return slide


def plot_weighting(slide_id,coords,weights,args,thumbnail_size=3000):
    print("Plotting final weights CHECK THESE ARE ACTUALLY FINAL for slide {}.".format(slide_id))

    slide = openslide.open_slide(args.data_slide_dir+"/"+slide_id+".svs")
    img = slide.get_thumbnail((thumbnail_size,thumbnail_size))
    plt.figure()
    plt.imshow(img)
    x_values, y_values = coords.T
    x_values=(x_values+128)*(thumbnail_size/max(slide.dimensions))
    y_values=(y_values+128)*(thumbnail_size/max(slide.dimensions))
    x_values=x_values.cpu()
    y_values=y_values.cpu()
    
    c='limegreen'
    c2='darkgreen'
    
    ## make it more transparent for lower values
    cmap = colors.LinearSegmentedColormap.from_list(
        'incr_alpha', [(0, (*colors.to_rgb(c),0)), (1, c2)])

    plt.scatter(x_values,y_values,c=weights,cmap=cmap,s=2, marker="s",edgecolors='none')
    plt.colorbar()
    plt.savefig('../mount_outputs/weight_maps/{}_{}.png'.format(slide_id,args.sampling_type), dpi=1000)
    plt.close()


def plot_weighting_gif(slide_id,sample_coords,coords,weights,args,iteration,slide=None,x_coords=None,y_coords=None,final_iteration=False,thumbnail_size=3000):
    if slide==None:
        slide = openslide.open_slide(args.data_slide_dir+"/"+slide_id+".svs")
        x_coords, y_coords = coords.T
        x_coords=(x_coords+128)*(thumbnail_size/max(slide.dimensions))
        y_coords=(y_coords+128)*(thumbnail_size/max(slide.dimensions))
        x_coords=x_coords.cpu()
        y_coords=y_coords.cpu()
    
    if iteration>0:
        img = slide.get_thumbnail((thumbnail_size,thumbnail_size))
        plt.figure()
        plt.imshow(img)
    
        c='limegreen'
        c2='darkgreen'

        ## make it more transparent for lower values
        cmap = colors.LinearSegmentedColormap.from_list(
            'incr_alpha', [(0, (*colors.to_rgb(c),0)), (1, c2)])
    
        plt.scatter(x_coords,y_coords,c=weights,cmap=cmap,s=2, marker="s",edgecolors='none')
        plt.colorbar()

        x_samples, y_samples = sample_coords.T
        x_samples=(x_samples+128)*(thumbnail_size/max(slide.dimensions))
        y_samples=(y_samples+128)*(thumbnail_size/max(slide.dimensions))
        x_samples=x_samples.cpu()
        y_samples=y_samples.cpu()
        plt.scatter(x_samples,y_samples,c='black',s=2,alpha=0.5,marker="s", edgecolors='none')

        plt.savefig('../mount_outputs/weight_maps/gifs/{}_{}_iter{}.png'.format(slide_id,args.sampling_type,iteration), dpi=500)
        plt.close()
    
    if final_iteration:
        print("Plotting weight gif for slide {} over {} iterations".format(slide_id,iteration+1))
        fp_in = "../mount_outputs/weight_maps/gifs/{}_{}_iter*.png".format(slide_id,args.sampling_type)
        fp_out = "../mount_outputs/weight_maps/{}_{}.gif".format(slide_id,args.sampling_type)
        imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
        img = next(imgs)  # extract first image from iterator
        img.save(fp=fp_out, format='GIF', append_images=imgs,save_all=True, duration=500, loop=1)

    return slide, x_coords, y_coords

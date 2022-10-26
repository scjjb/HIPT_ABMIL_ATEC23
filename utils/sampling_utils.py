import numpy as np

def generate_sample_idxs(idxs_length,previous_samples,sampling_weights,samples_per_epoch,num_random):
    nonrandom_idxs=list(np.random.choice(range(idxs_length),p=sampling_weights,size=int(samples_per_epoch-num_random),replace=False))
    previous_samples=previous_samples+nonrandom_idxs
    available_idxs=list(set(range(idxs_length))-set(previous_samples))
    random_idxs=list(np.random.choice(available_idxs, size=num_random,replace=False))
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


def update_sampling_weights(sampling_weights, attention_scores, all_sample_idxs, indices, neighbors, power=0.15, normalise = True, sampling_average = False, repeats_allowed = False):
    """
    Updates the sampling weights of all patches by looping through the most recent sample and adjusting all neighbors weights
    By default the weight of a patch is the maximum of its previous weight and the newly assigned weight, though sampling_average changes this to an average
    power is a hyperparameter controlling how attention scores are smoothed as typically very close to 0 or 1
    if repeated_allowed = False then weights for previous samples are set to 0
    """
    if sampling_average:
        for i in range(len(indices)):
            for index in indices[i][:neighbors]:
                if sampling_weights[index]>0:
                    sampling_weights[index]=(sampling_weights[index]+pow(attention_scores[i],power))/2
                else:
                    sampling_weights[index]=pow(attention_scores[i],power)
    else:
        for i in range(len(indices)):
            for index in indices[i][:neighbors]:
                sampling_weights[index]=max(sampling_weights[index],pow(attention_scores[i],power))

    if not repeats_allowed:
        for sample_idx in all_sample_idxs:
            sampling_weights[sample_idx]=0

    if normalise:
        sampling_weights=sampling_weights/max(sampling_weights)
        sampling_weights=sampling_weights/sum(sampling_weights)

    return sampling_weights

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

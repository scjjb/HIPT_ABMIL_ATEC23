import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score,balanced_accuracy_score
import numpy as np
import ast


parser = argparse.ArgumentParser(description='Model names input split by commas')
parser.add_argument('--model_names', type=str, default=None,help='models to plot')
parser.add_argument('--bootstraps', type=int, default=100000,
                    help='Number of bootstraps to calculate')
parser.add_argument('--run_repeats', type=int, default=10,
                            help='Number of model repeats')
parser.add_argument('--folds', type=int, default=10,
                            help='Number of cross-validation folds')
parser.add_argument('--data_csv', type=str, default='set_all_714.csv')
parser.add_argument('--label_dict',type=str,default="{'high_grade':0,'low_grade':1,'clear_cell':2,'endometrioid':3,'mucinous':4}")
parser.add_argument('--num_classes',type=int,default=2)
args = parser.parse_args()
model_names=args.model_names.split(",")
bootstraps=args.bootstraps
label_dict=ast.literal_eval(args.label_dict)

for model_name in model_names:
    model_name='eval_results/EVAL_'+model_name
    all_Ys=[]
    all_p1s=[]
    all_Yhats=[]
    #all_slides=[]
    all_ground_truths=[]
    ground_truths=pd.read_csv("dataset_csv/{}".format(args.data_csv))

    for run_no in range(args.run_repeats):
        Ys=[]
        p1s=[]
        Yhats=[]
        for fold_no in range(args.folds):
            if args.run_repeats>1:
                full_df = pd.read_csv(model_name+'_run{}/fold_{}.csv'.format(run_no,fold_no))
            else:
                full_df = pd.read_csv(model_name+'/fold_{}.csv'.format(fold_no))
            Ys=Ys+list(full_df['Y'])
            p1s=p1s+list(full_df['p_1'])
            Yhats=Yhats+list(full_df['Y_hat'])
            #all_slides=all_slides+list(full_df['slide_id'])
        all_Ys.append(Ys)
        all_p1s.append(p1s)
        all_Yhats.append(Yhats)

    f1s=[]
    accuracies=[]
    balanced_accuracies=[]
    for _ in range(bootstraps):
        idxs=np.random.choice(range(len(all_Ys)),len(all_Ys[0]))
        sample_Ys=[]
        sample_p1s=[]
        sample_Yhats=[]
        for i,idx in enumerate(idxs):
            sample_Ys=sample_Ys+[all_Ys[idx][i]]
            sample_p1s=sample_p1s+[all_p1s[idx][i]]
            sample_Yhats=sample_Yhats+[all_Yhats[idx][i]]
        if args.num_classes==2:
            f1s=f1s+[f1_score(sample_Ys,sample_Yhats)]
        else:
            f1s=f1s+[f1_score(sample_Ys,sample_Yhats,average='macro')]
        accuracies=accuracies+[accuracy_score(sample_Ys,sample_Yhats)]
        balanced_accuracies=balanced_accuracies+[balanced_accuracy_score(sample_Ys,sample_Yhats)]
    if args.num_classes==2:
        print("F1 mean: ",np.mean(f1s)," F1 std: ",np.std(f1s))
    else:
        print("Macro F1 mean: ",np.mean(f1s)," F1 std: ",np.std(f1s))
    print("accuracy mean: ",np.mean(accuracies)," accuracy std: ",np.std(accuracies))
    print("balanced accuracy mean: ",np.mean(balanced_accuracies)," balanced accuracy std: ",np.std(balanced_accuracies))

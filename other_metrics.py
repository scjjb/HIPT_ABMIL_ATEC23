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
    all_slides=[]
    all_ground_truths=[]
    ground_truths=pd.read_csv("dataset_csv/{}".format(args.data_csv))

    for run_no in range(args.run_repeats):
        for fold_no in range(args.folds):
            if args.run_repeats>1:
                full_df = pd.read_csv(model_name+'_run{}/fold_{}.csv'.format(run_no,fold_no))
            else:
                full_df = pd.read_csv(model_name+'/fold_{}.csv'.format(fold_no))
            all_Ys=all_Ys+list(full_df['Y'])
            all_p1s=all_p1s+list(full_df['p_1'])
            all_Yhats=all_Yhats+list(full_df['Y_hat'])
            #all_slides=all_slides+list(full_df['slide_id'])
    print("predicted")
    print("hgsc other")
    print(confusion_matrix(all_Ys,all_Yhats),"\n")
    
    ## below was a non-symettric confusion matrix shown as a 5x5 with lots of 0s
    #if args.num_classes<5:
        #for slide in all_slides:
        #    all_ground_truths=all_ground_truths+list(ground_truths.loc[ground_truths['slide_id'] == str(slide)]['label'])
        #    if len(ground_truths.loc[ground_truths['slide_id'] == str(slide)]['label'])>1:
        #        print(slide)
        #        print(ground_truths.loc[ground_truths['slide_id'] == str(slide)]['label'])
       #### print("predicted")
       #### print("hgsc other")
        #all_ground_truths=[label_dict[str(int(Y))] for Y in all_Ys]
       #### print(confusion_matrix(all_Ys,all_Yhats),"\n")
        #print(confusion_matrix([label_dict[true] for true in all_ground_truths],all_Yhats),"\n")

    f1s=[]
    accuracies=[]
    balanced_accuracies=[]
    for _ in range(bootstraps):
        idxs=np.random.choice(range(len(all_Ys)),len(all_Ys))
        if args.num_classes==2:
            f1s=f1s+[f1_score([all_Ys[idx] for idx in idxs],[all_Yhats[idx] for idx in idxs])]
        else:
            f1s=f1s+[f1_score([all_Ys[idx] for idx in idxs],[all_Yhats[idx] for idx in idxs],average='macro')]
        accuracies=accuracies+[accuracy_score([all_Ys[idx] for idx in idxs],[all_Yhats[idx] for idx in idxs])]
        balanced_accuracies=balanced_accuracies+[balanced_accuracy_score([all_Ys[idx] for idx in idxs],[all_Yhats[idx] for idx in idxs])]
    if args.num_classes==2:
        print("F1 mean: ",np.mean(f1s)," F1 std: ",np.std(f1s))
    else:
        print("Macro F1 mean: ",np.mean(f1s)," F1 std: ",np.std(f1s))
    print("accuracy mean: ",np.mean(accuracies)," accuracy std: ",np.std(accuracies))
    print("balanced accuracy mean: ",np.mean(balanced_accuracies)," balanced accuracy std: ",np.std(balanced_accuracies))

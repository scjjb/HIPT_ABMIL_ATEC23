import argparse
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np

def calculate_error(Y_hat, Y):
    if Y_hat==Y:
        error=0
    else:
        error=1
    return error



parser = argparse.ArgumentParser(description='Model names input split by commas')
parser.add_argument('--model_names', type=str, default=None,help='models to plot')
parser.add_argument('--bootstraps', type=int, default=100000,
                    help='Number of bootstraps to calculate')
parser.add_argument('--run_repeats', type=int, default=10,
                            help='Number of model repeats')
parser.add_argument('--folds', type=int, default=10,
                            help='Number of cross-validation folds')
parser.add_argument('--num_classes',type=int,default=2,help='Number of classes')
args = parser.parse_args()
model_names=args.model_names.split(",")
bootstraps=args.bootstraps

for model_name in model_names:
    model_name='eval_results/EVAL_'+model_name
    all_Ys=[]
    all_p1s=[]
    all_probs=[]
    all_Yhats=[]
    for run_no in range(args.run_repeats):
        for fold_no in range(args.folds):
            if args.run_repeats>1:
                full_df = pd.read_csv(model_name+'_run{}/fold_{}.csv'.format(run_no,fold_no+2))
            else:
                full_df = pd.read_csv(model_name+'/fold_{}.csv'.format(fold_no))
            all_Ys=all_Ys+list(full_df['Y'])
            if args.num_classes==2:
                all_p1s=all_p1s+list(full_df['p_1'])
            else:
                if len(all_probs)<1:
                    all_probs=full_df.iloc[:,-args.num_classes:]
                else:
                    all_probs=all_probs.append(full_df.iloc[:,-args.num_classes:])
            all_Yhats=all_Yhats+list(full_df['Y_hat'])


    AUC_scores=[]
    err_scores=[]
    for _ in range(bootstraps):
        idxs=np.random.choice(range(len(all_Ys)),len(all_Ys))
        if args.num_classes>2:
            #print("len probs(0)",len(all_probs[0]))
            #print(all_probs[0])
            #print(all_probs)
            #print("len probs",len(all_probs))
            #print("len Ys",len(all_Ys))
            AUC_scores=AUC_scores+[roc_auc_score([all_Ys[idx] for idx in idxs],[all_probs.iloc[idx,:] for idx in idxs],multi_class='ovr')]
        else:
            AUC_scores=AUC_scores+[roc_auc_score([all_Ys[idx] for idx in idxs],[all_p1s[idx] for idx in idxs])]
        error=0
        for idx in idxs:
            error=error+calculate_error(all_Yhats[idx],all_Ys[idx])
        error=error/len(idxs)
        err_scores=err_scores+[error]

print("AUC mean: ",np.mean(AUC_scores)," AUC std: ",np.std(AUC_scores))
print("Acc mean: ",1-np.mean(err_scores), "Acc std: ",np.std(err_scores))


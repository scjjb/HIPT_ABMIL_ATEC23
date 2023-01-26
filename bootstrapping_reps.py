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
        Ys=[]
        probs=[]
        p1s=[]
        Yhats=[]
        for fold_no in range(args.folds):
            if args.run_repeats>1:
                full_df = pd.read_csv(model_name+'_run{}/fold_{}.csv'.format(run_no,fold_no))
            else:
                full_df = pd.read_csv(model_name+'/fold_{}.csv'.format(fold_no))
            Ys=Ys+list(full_df['Y'])
            if args.num_classes==2:
                p1s=p1s+list(full_df['p_1'])
            else:
                if len(all_probs)<1:
                    probs=full_df.iloc[:,-args.num_classes:]
                else:
                    probs=probs.append(full_df.iloc[:,-args.num_classes:])
            Yhats=Yhats+list(full_df['Y_hat'])
        all_Ys.append(Ys)
        all_probs.append(probs)
        all_p1s.append(p1s)
        all_Yhats.append(Yhats)

    AUC_scores=[]
    err_scores=[]
    for _ in range(bootstraps):
        idxs=np.random.choice(range(len(all_Ys)),len(all_Ys[0]))
        sample_Ys=[]
        sample_probs=[]
        sample_p1s=[]
        for i,idx in enumerate(idxs):
            sample_Ys=sample_Ys+[all_Ys[idx][i]]
            if args.num_classes>2:
                sample_probs=sample_probs+[all_probs[idx][i]]
            else:
                sample_p1s=sample_p1s+[all_p1s[idx][i]]
        if args.num_classes>2:
            AUC_scores=AUC_scores+[roc_auc_score(sample_Ys,sample_probs,multi_class='ovr')]
        else:
            AUC_scores=AUC_scores+[roc_auc_score(sample_Ys,sample_p1s)]
        error=0

print("AUC mean: ",np.mean(AUC_scores)," AUC std: ",np.std(AUC_scores))
    




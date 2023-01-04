from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from matplotlib.font_manager import FontProperties

parser = argparse.ArgumentParser(description='Model names input split by commas')
parser.add_argument('--model_names', type=str, default=None,help='models to plot')
parser.add_argument('--run_repeats', type=int, default=10,
                                    help='Number of model repeats')
parser.add_argument('--folds', type=int, default=10,
                                    help='Number of cross-validation folds')
parser.add_argument('--num_classes',type=int,default=2,help='Number of classes')
args = parser.parse_args()
model_names=args.model_names.split(",")

font_legend = FontProperties()
#font_legend.addfont('Times New Roman')
font_legend.set_family('serif')
font_legend.set_name('Times New Roman')

font = {'fontname':'Times New Roman'}

for model_name in model_names:
    full_model_name='eval_results/EVAL_'+model_name

    for run_no in range(args.run_repeats):
        all_Ys=[]
        all_p1s=[]
        for fold_no in range(args.folds):
            if args.run_repeats>1:
                full_df = pd.read_csv(full_model_name+'_run{}/fold_{}.csv'.format(run_no,fold_no))
            else:
                full_df = pd.read_csv(full_model_name+'/fold_{}.csv'.format(fold_no))
            all_Ys=all_Ys+list(full_df['Y'])
            all_p1s=all_p1s+list(full_df['p_1'])

        fpr, tpr, threshold = roc_curve(all_Ys, all_p1s)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label = 'Repeat '+str(run_no+1))

    #plt.title('Receiver Operating Characteristic',**font)
    plt.legend(loc = 'lower right',prop=font_legend)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks(**font)
    plt.yticks(**font)
    plt.ylabel('True Positive Rate',**font)
    plt.xlabel('False Positive Rate',**font)
    plt.show()

    plt.savefig('roc_plots/{}.png'.format(model_name),dpi=300)

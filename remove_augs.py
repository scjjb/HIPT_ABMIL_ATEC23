import pandas as pd
import argparse
import os
parser = argparse.ArgumentParser(description='Script to remove augmentations from training and validation folds')
parser.add_argument('--split_name', type=str, default=None)
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
args = parser.parse_args()

os.mkdir('splits/{}_trainaugsonly'.format(args.split_name))

for i in range(args.k):
    df = pd.read_csv('splits/{}/splits_{}.csv'.format(args.split_name,i),keep_default_na=False)
    train=df['train']
    val=df['val'][~df['val'].str.contains("aug")]
    test=df['test'][~df['test'].str.contains("aug")]
    print(train)
    print(val)
    
    dic={"train": train, "val": val, "test": test}
    new_df=pd.DataFrame(dic)
    new_df=new_df.fillna('')
    new_df.reset_index(drop=True,inplace=True)
    print(new_df)
    new_df.to_csv('splits/{}_trainaugsonly/splits_{}.csv'.format(args.split_name,i))

import pandas as pd

for i in range(4):
    df = pd.read_csv('splits/custom_1vsall_998_aug_100/splits_{}.csv'.format(i),keep_default_na=False)
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
    new_df.to_csv('splits/custom_1vsall_trainpartaug_998_100_4fold/splits_{}.csv'.format(i))

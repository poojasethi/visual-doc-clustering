import pandas as pd
import json
from datasets import Dataset, concatenate_datasets
import numpy as np

# Bring in data
a_file = open("data/encodings/layoutlmv2_ft_labels_dict_final.json", "r")
label2idx = json.loads(a_file.read())
a_file.close()

def resolve_issues(df):
    df["bbox"] = df["bbox"].apply(lambda x: np.array(x).flatten())
    df["image"] = df["image"].apply(lambda x: np.array(x).flatten())
    
    dt = Dataset.from_pandas(df)
    return dt

df0 = resolve_issues(pd.read_pickle('data/encodings/layoutlmv2_ft_encodings0.pkl'))
df1 = resolve_issues(pd.read_pickle('data/encodings/layoutlmv2_ft_encodings1.pkl'))
df2 = resolve_issues(pd.read_pickle('data/encodings/layoutlmv2_ft_encodings2.pkl'))
df3 = resolve_issues(pd.read_pickle('data/encodings/layoutlmv2_ft_encodings3.pkl'))
df4 = resolve_issues(pd.read_pickle('data/encodings/layoutlmv2_ft_encodings4.pkl'))
df5 = resolve_issues(pd.read_pickle('data/encodings/layoutlmv2_ft_encodings5.pkl'))
df6 = resolve_issues(pd.read_pickle('data/encodings/layoutlmv2_ft_encodings6.pkl'))

df = concatenate_datasets([df0, df1, df2, df3, df4, df5, df6]).to_pandas()

df['label_idx'] = [label2idx[x] for x in df['label']]

print(df[['label', 'label_idx']])
print(df['label_idx'].unique())
print(len(df))

df.to_pickle('data/encodings/layoutlmv2_ft_encodings.pkl')
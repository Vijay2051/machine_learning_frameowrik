from os import name
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    df = pd.read_csv("input/train.csv")
    df['kfold'] = -1

    df = df.sample(frac=1).reset_index(drop=True)
    kf = StratifiedKFold(n_splits=5, shuffle = False, random_state=42)

    for folds, (train_idx, test_idx) in enumerate(kf.split(X=df, y=df.target.values)):
        print("train_idx : ", len(train_idx), "test_idx :", len(test_idx))
        df.loc[test_idx ,"kfold"] = folds

    df.to_csv("input/train_folds.csv", index=False)


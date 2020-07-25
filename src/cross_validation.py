import pandas as pd
from sklearn import model_selection

"Types of classification problems"
"""
    - binary classification
    - multi class classification
    - multi label classification
    - single column regression
    - multi column regression
    - holdout 

"""



class Cross_validation:
    def __init__(
        self,
        df, 
        target_cols, 
        problem_type = "binary_classification",
        num_folds = 5,
        shuffle = True,
        random_state = 42
    ):
        self.dataframe = df
        self.target_cols = target_cols
        self.num_target = len(target_cols)
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.shuffle = shuffle
        self.random_state = random_state

        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

        self.dataframe['kfold'] = -1


    def split(self):
        if self.problem_type in ["binary_classification", "multiclass_classification"]:
            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique()
            if unique_values == 1:
                raise Exception("Only one target value is present")
            elif unique_values >1:
                kf = model_selection.StratifiedKFold(n_splits=self.num_folds, shuffle = False) #here random_state is not necessary coz the shuffle is False, it will raise a warning
                for folds, (train_idx, valid_idx) in enumerate(kf.split(X = self.dataframe, y=self.dataframe[target].values)):
                    print("train_idx : ", len(train_idx), "test_idx :", len(valid_idx))
                    self.dataframe.loc[valid_idx, "kfold"] = folds
        print(self.dataframe)
        return self.dataframe

if __name__ == "__main__":
    df = pd.read_csv("../input/train.csv")
    cv = Cross_validation(df, target_cols=['target'])
    df_split = cv.split()

        

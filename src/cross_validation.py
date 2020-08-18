import pandas as pd
from sklearn import model_selection

"Types of classification problems"
"""
    - binary classification
    - multi class classification
    - multi label classification
    - single_column_regression
    - multi_column_regression
    - holdout 

"""



class Cross_validation:
    def __init__(
        self,
        df, 
        target_cols, 
        problem_type = "binary_classification",
        mulitlabel_delimiter = ",",
        num_folds = 5,
        shuffle = True,
        random_state = 42
    ):
        self.dataframe = df
        self.target_cols = target_cols
        self.num_target = len(target_cols)
        self.problem_type = problem_type
        self.mulitlabel_delimiter = mulitlabel_delimiter
        self.num_folds = num_folds
        self.shuffle = shuffle
        self.random_state = random_state

        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

        self.dataframe['kfold'] = -1


    def split(self):
        if self.problem_type in ("binary_classification", "multiclass_classification"):
            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique()
            if unique_values == 1:
                raise Exception("Only one target value is present")
            elif unique_values >1:
                kf = model_selection.StratifiedKFold(n_splits=self.num_folds, shuffle = False) #here random_state is not necessary coz the shuffle is False, it will raise a warning
                for folds, (train_idx, valid_idx) in enumerate(kf.split(X = self.dataframe, y=self.dataframe[target].values)):
                    print("train_idx : ", len(train_idx), "test_idx :", len(valid_idx))
                    self.dataframe.loc[valid_idx, "kfold"] = folds

        elif self.problem_type in ("single_column_regression", "multi_column_regression"):
            if self.num_target != 1 and self.problem_type == "single_column_regression":
                raise Exception ("Invalid number of probelm for this problem type")
            if self.num_target <2 and self.problem_type == "multi_column_regression":
                raise Exception ("invalid number of targets for this problem type")
            kf = model_selection.KFold(n_splits = self.num_folds)
            for folds, (train_idx, valid_idx) in enumerate(kf.split(X = self.dataframe)):
                self.dataframe.loc[valid_idx, "kfold"] = folds

            
        elif self.problem_type.startswith("holdout_"):
            holdout_percentage = int(self.problem_type.split("_")[1])
            num_holdout_samples = int(len(self.dataframe) * holdout_percentage/100)
            self.dataframe.loc[:len(self.dataframe) - num_holdout_samples, "kfold"] = 1
            self.dataframe.loc[len(self.dataframe) - num_holdout_samples:, "kfold"] = 0

        
        elif self.problem_type == "multilabel_classification":
            if self.num_target != 1:
                raise Exception("invalid number of targets for this problem type")
            targets = self.dataframe[self.target_cols[0]].apply(lambda x :  len(str(x).split(self.mulitlabel_delimiter)))
            kf = model_selection.StratifiedKFold(n_splits=self.num_folds)
            for folds, (train_idx, valid_idx) in enumerate(kf.split(X = self.dataframe, y=targets)):
                self.dataframe.loc[valid_idx, "kfold"] == folds


        else:
            raise Exception("Problem type is not understood")

        

        print(self.dataframe)
        return self.dataframe

if __name__ == "__main__":
    df = pd.read_csv("../input/train.csv")
    cv = Cross_validation(df, target_cols=['target'],  problem_type="holdout_10")
    df_split = cv.split()
    print(df_split.head())
    print(df_split['kfold'].value_counts()) 
        

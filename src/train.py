import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
from . import dispatcher
import joblib

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")

FOLD_MAPPING = {
    0 : [1,2,3,4],
    1 : [0,2,3,4],
    2 : [0,1,3,4],
    3 : [0,1,2,4],
    4 : [0,1,2,3]
}

if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold == FOLD]

    y_train = train_df.target.values
    y_valid = valid_df.target.values


    train_df = train_df.drop(['id', 'target', 'kfold'], axis=1)
    valid_df = valid_df.drop(['id', 'target', 'kfold'], axis=1)

    label_encoders = {}

    print(len(train_df.columns))
    print(len(valid_df.columns))

    valid_df = valid_df[train_df.columns]

    for columns in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        train_df.loc[:, columns] = train_df.loc[:, columns].astyp e(    str).fillna("NONE")
        valid_df.loc[:, columns] = valid_df.loc[:, columns].astype(str).fillna("NONE")
        df_test.loc[:, columns] = df_test.loc[:, columns].astype(str).fillna("NONE")

        lbl.fit(train_df[columns].values.tolist() + valid_df[columns].values.tolist() + df_test[columns].values.tolist()) 
        train_df.loc[:, columns] = lbl.transform(train_df[columns].values.tolist())
        valid_df.loc[:, columns] = lbl.transform(valid_df[columns].values.tolist())
        label_encoders[columns] = lbl


    # Data is now ready to train
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, y_train)
    predi = clf.predict_proba(valid_df)[:, 1]
    print(metrics.roc_auc_score(y_valid, predi))

    joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_label_enocders.pkl")
    joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")
    
    
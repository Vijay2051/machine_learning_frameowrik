"""Frst create the files that needed for the ML framework to be created"""

    # create a folder where the files have to be created

    # create files nammely
        # __init__.py
        # train.py
        # metrics.py
        # create_folds.py
        # predict.py
        # dataset.py
        # loss.py
        # utils.py
        # feature_gen.py
        # dispatcher.py
        # engine.py

    """
        the next thing is to create the file for create_folds.py
    """

    Import the Stratifiedkfold from the scikit learn
    then create a new column with name kfold and place the values with -1
    then create a stratified classifier
    enumerate the train_idx and test_idx, folds with the classifier.split(X = df, y=df.target.values)
    and replace the test data set created columns of kfolds with the enumearted value
    then export the csv as train_folds.csv into the input folder

        # These are the things that need to be done in the create_folds.py



    """
        The next thing to do is the train.py
        Where we will be importing the train_folds.csv
        and creating the train and test datasets from this
    """

    we will br creating two datsets train and valid data sets
    then we will be working on the model creation and training the model
    and then storing these trained models and lable_encoders in the MOdels folder

    


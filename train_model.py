import argparse 
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

from ml.config import load_config, get_model_name_from_cfg

from ml.paths import APP_ROOT, DATA_DIR, MODEL_DIR



def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model using the provided YAML configuration file."
    )
    parser.add_argument(
        "--config",
        required=False,
        help="Path to the YAML configuration file (e.g. configs/random_forest.yaml)",
    )
    parser.add_argument(
        "--model",
        required=False,
        help="Path to the YAML configuration file (e.g. configs/random_forest.yaml)",
    )
    return parser.parse_args()




def main():

    # Parse Arguements
    args = parse_args()

    # Create empy Config
    cfg = {}

    # Check config for arguments
    if args.config and args.config.strip():
        cfg = load_config(args.config)
        print(f"Configuration Found and Loaded: {cfg}")
    else:
        cfg = load_config("configs/random_forest.yaml")
        print(f"No configuration file was provided, using default - Random Forest Classifier")


    print(cfg)

    model_cfg = cfg.get("model", {})
    print(model_cfg)

    train_cfg = cfg.get("train", {})
    print(train_cfg)

    model_name = get_model_name_from_cfg(cfg)
    print(model_name)



    # TODO: load the cencus.csv data
    #project_path = "Your path here"
    #project_path = DATA_DIR
    #data_path = os.path.join(project_path, "data", "census.csv")

    project_path = APP_ROOT
    print(project_path)

    data_path = os.path.join(DATA_DIR, "census.csv")
    print(data_path)

    #data = None # your code here
    data = pd.read_csv(data_path)

    data.head()

    # TODO: split the provided data to have a train dataset and a test dataset
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = None, None# Your code here

    # DO NOT MODIFY
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    '''
    # TODO: use the process_data function provided to process the data.
    X_train, y_train, encoder, lb = process_data(
        # your code here
        # use the train dataset 
        # use training=True
        # do not need to pass encoder and lb as input
        )

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    '''
    # TODO: use the train_model function to train the model on the training dataset
    model = None # your code here

    # save the model and the encoder
    #model_path = os.path.join(project_path, "model", "model.pkl")
    #save_model(model, model_path)

    model_path = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")
    print(model_path)



    #encoder_path = os.path.join(project_path, "model", "encoder.pkl")
    #save_model(encoder, encoder_path)

    encoder_path = os.path.join(MODEL_DIR, f"{model_name}_encoder.pkl")
    print(encoder_path)

    # load the model
    #model = load_model(
        #model_path
    #) 

    # TODO: use the inference function to run the model inferences on the test dataset.
    preds = None # your code here

    '''
    # Calculate and print the metrics
    p, r, fb = compute_model_metrics(y_test, preds)
    print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

    # TODO: compute the performance on model slices using the performance_on_categorical_slice function
    # iterate through the categorical features
    for col in cat_features:
        # iterate through the unique values in one categorical feature
        for slicevalue in sorted(test[col].unique()):
            count = test[test[col] == slicevalue].shape[0]
            p, r, fb = performance_on_categorical_slice(
                # your code here
                # use test, col and slicevalue as part of the input
            )
            with open("slice_output.txt", "a") as f:
                print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
                print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)
    
    '''

if __name__ == "__main__":
    main()
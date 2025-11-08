import argparse 
import os
from pathlib import Path
import logging

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

from ml.config import load_config, get_model_name_from_cfg, parse_cfg

from ml.paths import APP_ROOT, DATA_DIR, MODEL_DIR



logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Train a model using the provided YAML configuration file."
    )

    # Configs
    parser.add_argument(
        "-c", 
        "--config",
        required=False,
        help="Path to the YAML configuration file (e.g. configs/random_forest.yaml)",
    )

     # Data Files
    parser.add_argument(
        "--data-dir", 
        help="Override io.data_dir"
    )
    parser.add_argument(
        "--data-file", 
        help="Override io.data_file"
    )
    parser.add_argument(
        "--data-path", 
        help="Direct file path (bypass data_dir/data_file)"
    )

    return parser.parse_args()


def apply_cli_overrides(cfg, args):
    
    io_cfg = cfg.setdefault("io", {})

    # Data Files
    if args.data_dir:  io_cfg["data_dir"]  = args.data_dir
    if args.data_file: io_cfg["data_file"] = args.data_file
    if args.data_path: io_cfg["data_path"] = args.data_path

    return cfg


def load_data(cfg):

    io_cfg = cfg.get("io", {}) if cfg else {}

    data_path = DATA_DIR
    data_filename = io_cfg.get("data_file")

    data_file_path = os.path.join(data_path, data_filename)
    print(f"Date File Path: {data_file_path}")

    #data = None # your code here
    data = pd.read_csv(data_file_path)

    return data



def resolve_model_path(cfg):
    io_cfg = cfg.get("io", {}) if cfg else {}

    # Default model name
    model_name = io_cfg.get("model_name") or get_model_name_from_cfg(cfg)

    # If a subdirectory is specified, use it — otherwise, save directly under MODEL_DIR/model_name
    model_subdir = io_cfg.get("model_subdir")

    if model_subdir:
        save_dir = (MODEL_DIR / model_subdir).resolve()
    else:
        save_dir = (MODEL_DIR / model_name).resolve()

    save_dir.mkdir(parents=True, exist_ok=True)

    return save_dir, model_name
    

def main():

    # Parse Arguements
    args = parse_args()

    # Create empy Config
    cfg = {}

    # Load Config via arguments
    if args.config and args.config.strip():
        cfg = load_config(args.config)
        print(f"Configuration Found and Loaded: {cfg}")
    else:
        cfg = load_config("configs/random_forest.yaml")
        print(f"No configuration file was provided, using default - Random Forest Classifier")

    cfg = apply_cli_overrides(cfg, args)

    model_cfg, train_cfg, io_cfg = parse_cfg(cfg)
    
    print()
    print(model_cfg)
    print()
    print(train_cfg)
    print()
    print(io_cfg)
    print()


    save_dir, model_name = resolve_model_path(cfg)
    
    print(f"Save directory: {save_dir}")
    print(f"Model Name: {model_name}")


    # TODO: load the cencus.csv data
    #project_path = "Your path here"
    #project_path = DATA_DIR
    #data_path = os.path.join(project_path, "data", "census.csv")


    data = load_data(cfg)

    data.head()
    data.info()


    # TODO: split the provided data to have a train dataset and a test dataset
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    #train, test = None, None# Your code here

    #X = data.drop(train_cfg.get("target"), axis=1)
    #print(X.info())
    #print(X.head(5))
    #y = data[train_cfg.get("target")]
    #print(y.info())
    #print(y.head(5))

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_cfg.get("test_size"), random_state=train_cfg.get("random_state"))

    #print(X_train.info())
    #print(X_test.info())


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

    train_df, holdout_df = train_test_split(data, test_size=train_cfg.get("holdout_size"), stratify=data[train_cfg.get("target")], random_state=train_cfg.get("random_state"))



    def data_split_kfold(data, cfg):

        model_cfg, train_cfg, io_cfg = parse_cfg(cfg)

        fold_metrics = []

        skf = StratifiedKFold(n_splits=train_cfg.get("cv_folds"), shuffle=True, random_state=train_cfg.get("random_state"))

        for fold, (train_idx, val_idx) in enumerate(skf.split(data, data[train_cfg.get("target")]), 1):

            train_df = data.iloc[train_idx]
            val_df = data.iloc[val_idx]

            X_train, y_train, encoder, label_binarizer = process_data(
                train_df,
                categorical_features=cat_features,
                label=train_cfg.get("target"),
                training = True
            )

            X_val, y_val, _, _ = process_data(
                val_df,
                categorical_features=cat_features,
                label=train_cfg.get("target"),
                training = False,
                encoder = encoder,
                lb = label_binarizer
            )

            model = train_model(X_train, y_train, cfg=cfg)

            preds = inference(model, X_val, proba=False)

            precision, recall, fbeta = compute_model_metrics(y_val, preds)

            print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | Fβ: {fbeta:.4f}")

            fold_metrics.append((precision, recall, fbeta))

        fold_metrics = np.array(fold_metrics)
        
        mean_precision, mean_recall, mean_fbeta = np.mean(fold_metrics, axis=0)

        print("\n=== Cross-validation Summary ===")
        print(f"Mean Precision: {mean_precision:.4f}")
        print(f"Mean Recall:    {mean_recall:.4f}")
        print(f"Mean Fβ:        {mean_fbeta:.4f}")

        return mean_precision, mean_recall, mean_fbeta 


    mean_precision, mean_recall, mean_fbeta = data_split_kfold(train_df, cfg)
    

    X_full, y_full, encoder, label_binarizer = process_data(
        train_df,
        categorical_features = cat_features,
        label = train_cfg.get("target"),
        training = True
    )

    model = train_model(X_full, y_full, cfg=cfg)

    predictions_full = inference(model, X_full, proba=False)

    precision_full, recall_full, fbeta_full = compute_model_metrics(y_full, predictions_full)

    final_metrics = {"precision": float(precision_full), "recall": float(recall_full), "fbeta": float(fbeta_full)}

    print(final_metrics)

    params = model_cfg.get("params")
    

    save_model(
        model=model,
        encoder=encoder,
        label_binarizer=label_binarizer,
        cfg=cfg,
        metrics=final_metrics,
        parameters=params,
        save_dir=save_dir,
        model_name = model_name
)


    '''
    # TODO: use the process_data function provided to process the data.
    X_train, y_train, encoder, lb = process_data(
        # your code here
        # use the train dataset 
        # use training=True
        # do not need to pass encoder and lb as input
        )
    
    X_train, y_train, encoder, lb = process_data(
        X_train_df,
        categorical_features=cat_features,
        label=train_cfg.get("target"),
        training=True
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
    #model = None # your code here
    #model = train_model(X_train, y_train, cfg=cfg)

    # save the model and the encoder
    #model_path = os.path.join(project_path, "model", "model.pkl")
    #save_model(model, model_path)

    #model_path = os.path.join(MODEL_DIR, cfg.get("model_subdir"), f"{model_name}_model.pkl")
    #print(model_path)



    #encoder_path = os.path.join(project_path, "model", "encoder.pkl")
    #save_model(encoder, encoder_path)

    #encoder_path = os.path.join(MODEL_DIR, cfg.get("model_subdir"), f"{model_name}_encoder.pkl")
    #print(encoder_path)

    # load the model


    #model = load_model()

    # TODO: use the inference function to run the model inferences on the test dataset.
    #preds = None # your code here


    # Calculate and print the metrics
    #p, r, fb = compute_model_metrics(y_test, preds)
    #print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

    

    # TODO: compute the performance on model slices using the performance_on_categorical_slice function
    # iterate through the categorical features
    for col in cat_features:
        # iterate through the unique values in one categorical feature
        #for slicevalue in sorted(test[col].unique()):
        for slicevalue in sorted(holdout_df[col].unique()):
            count = int((holdout_df[col] == slicevalue).sum())
            
            p, r, fb = performance_on_categorical_slice(
                # your code here
                # use test, col and slicevalue as part of the input
                
                data=holdout_df,
                column_name=col,
                slice_value = slicevalue,
                categorical_features=cat_features,
                label=train_cfg.get("target"),
                encoder=encoder,
                label_binarizer=label_binarizer,
                model=model

            )
            with open("slice_output.txt", "a") as f:
                print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
                print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)
    

if __name__ == "__main__":
    main()
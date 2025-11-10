import argparse 
import os
from pathlib import Path

import numpy as np
import pandas as pd

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    load_model_full,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

from ml.config import load_config, get_model_name_from_cfg, parse_cfg
from ml.paths import APP_ROOT, DATA_DIR, MODELS_DIR


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
    logger.info(f"Date File Path: {data_file_path}")

    #data = None # your code here
    data = pd.read_csv(data_file_path)

    return data



def resolve_model_path(cfg):
    io_cfg = cfg.get("io", {}) if cfg else {}

    # Default model name
    model_name = io_cfg.get("model_name") or get_model_name_from_cfg(cfg)

    # If a subdirectory is specified, use it — otherwise, save directly under MODELS_DIR/model_name
    model_subdir = io_cfg.get("model_subdir")

    if model_subdir:
        save_dir = (MODELS_DIR / model_subdir).resolve()
    else:
        save_dir = (MODELS_DIR / model_name).resolve()

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
        logger.info(f"Configuration Found and Loaded: {cfg}")
    else:
        cfg = load_config("configs/random_forest.yaml")
        logger.info(f"No configuration file was provided, using default - Random Forest Classifier")

    cfg = apply_cli_overrides(cfg, args)

    model_cfg, train_cfg, io_cfg = parse_cfg(cfg)
    
    logger.info(model_cfg)
    logger.info(train_cfg)
    logger.info(io_cfg)


    save_dir, model_name = resolve_model_path(cfg)
    
    logger.info(f"Save directory: {save_dir}")
    logger.info(f"Model Name: {model_name}")


    data = load_data(cfg)

    data.head()
    data.info()


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

            logger.info(f"Precision: {precision:.4f} | Recall: {recall:.4f} | Fβ: {fbeta:.4f}")

            fold_metrics.append((precision, recall, fbeta))

        fold_metrics = np.array(fold_metrics)
        
        mean_precision, mean_recall, mean_fbeta = np.mean(fold_metrics, axis=0)

        logger.info("\n=== Cross-validation Summary ===")
        logger.info(f"Mean Precision: {mean_precision:.4f}")
        logger.info(f"Mean Recall:    {mean_recall:.4f}")
        logger.info(f"Mean Fβ:        {mean_fbeta:.4f}")

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

    logger.info(final_metrics)

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

    model, encoder, label_binarizer = load_model_full(model_name, cfg)

    logger.info(type(model))
    logger.info(type(encoder))
    logger.info(type(label_binarizer))

    logger.info(hasattr(model, "fit"))  # should be True
    logger.info(hasattr(model, "predict"))  # should be True
    logger.info(hasattr(model, "n_features_in_"))  # confirms it was trained/fitted
    logger.info(hasattr(encoder, "categories_"))  # confirms fitted encoder
    logger.info(hasattr(label_binarizer, "classes_"))  # confirms fitted binarizer
    


    # iterate through the categorical features
    for col in cat_features:
        # iterate through the unique values in one categorical feature
        #for slicevalue in sorted(test[col].unique()):
        for slicevalue in sorted(holdout_df[col].unique()):
            count = int((holdout_df[col] == slicevalue).sum())
            
            p, r, fb = performance_on_categorical_slice(                
                data=holdout_df,
                column_name=col,
                slice_value = slicevalue,
                categorical_features=cat_features,
                label=train_cfg.get("target"),
                encoder=encoder,
                label_binarizer=label_binarizer,
                model=model
            )

            slice_output_filename = f"{model_name}_slice_output.txt"

            slice_output_filepath = save_dir / slice_output_filename

            logger.info(f"Printing to output file: {col}: {slicevalue}, Count: {count:,}")
            logger.info(f" Printing to output file: Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

            with open(slice_output_filepath, "a") as f:
                print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
                print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)
    

if __name__ == "__main__":
    main()
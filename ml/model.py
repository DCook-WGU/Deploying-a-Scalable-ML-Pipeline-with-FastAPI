import pickle
import json 
import importlib 
from typing import Any, Dict, Optional
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data
# TODO: add necessary import

from sklearn.ensemble import RandomForestClassifier
from ml.paths import APP_ROOT, DATA_DIR, MODEL_DIR


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def _filter_params_for_cls(cls, params: Dict[str, Any]) -> Dict[str, Any]:

    try:
        inst = cls()  
        valid = set(getattr(inst, "get_params")().keys())
        return {key: value for key, value in params.items() if key in valid}
    except Exception:
        return params  

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def _build_estimator(
    cfg: Optional[Dict[str, Any]] = None, 
    class_path: Optional[str] = None, 
    parameters: Optional[Dict[str, Any]] = None, 
    default_random_state: int = 42):

    if cfg:
        model_cfg = cfg.get("model", {}) or {}
        train_cfg = cfg.get("train", {}) or {}

        class_path = class_path or model_cfg.get("class_path")
        parameters = parameters or model_cfg.get("params", {})
        default_random_state = train_cfg.get("random_state", default_random_state)

    if not class_path:
        class_path = "sklearn.ensemble.RandomForestClassifier"

    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
    except(ImportError, AttributeError, ValueError) as e:
        raise ImportError(f"Could not import model: '{class_path}':{e}")

    parameters = dict(parameters or {})

    if "random_state" not in parameters:
        try:
            cls(random_state=default_random_state)
            parameters["random_state"] = default_random_state
        except TypeError:
            Pass

    parameters = _filter_params_for_cls(cls, parameters)

    return cls(**parameters)




#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def _add_suffixes(cfg):

    defaults = {
        "model": "_model.pkl",
        "encoder": "_encoder.pkl",
        "label_binarizer": "_label_binarizer.pkl",
        "params": "params.json",
        "metrics": "metrics.json",
    }

    return {**defaults, **(cfg.get("io", {}).get("file_name_suffixes", {}) )}


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 



# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, cfg):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = _build_estimator(cfg)
    model.fit(X_train, y_train)

    return model





#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """

    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)

    return precision, recall, fbeta


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def inference(model, X, proba=False, threshold=0.5):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    if proba:
        probs = model.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int), probs
    else:
        return model.predict(X)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def save_model(model, encoder, label_binarizer, cfg, metrics, parameters, save_dir, model_name):
    """ Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """


    io_cfg = (cfg.get("io") or {})

    save_dir = save_dir

    base_filename = model_name or io_cfg.get("model_name", "model")

    suffixes = (io_cfg.get("file_name_suffixes") or {})
    
    suffix_model = suffixes.get("model", "_model.pkl")
    suffix_encoder = suffixes.get("encoder", "_encoder.pkl")
    suffix_label_binarizer = suffixes.get("label_binarizer", "_label_binarizer.pkl")
    suffix_metrics = suffixes.get("metrics", "_metrics.json")
    suffix_parameters = suffixes.get("parameters", "_params.json")

    model_file_path = save_dir / f"{base_filename}{suffix_model}"
    encoder_file_path = save_dir / f"{base_filename}{suffix_encoder}"
    label_binarizer_file_path = save_dir / f"{base_filename}{suffix_label_binarizer}"
    metrics_file_path = save_dir / f"{base_filename}{suffix_metrics}"
    parameters_file_path = save_dir / f"{base_filename}{suffix_parameters}"

    allow_overwrite  = bool(io_cfg.get("allow_overwrite", True))

    files_to_check_for = [model_file_path, encoder_file_path, label_binarizer_file_path, metrics_file_path, parameters_file_path]

    existing_files_list = []

    for file in files_to_check_for:
        if file.exists():
            existing_files_list.append(file)

    if existing_files_list and not allow_overwrite :

        resolved_existing_files_list = []
        
        for file in existing_files_list:
            absolute_path = file.resolve()
            absolute_path_string = str(absolute_path)
            resolved_existing_files_list.append(absolute_path_string)

        existing_files_list_str = "\n - ".join(resolved_existing_files_list)
        
        # One line variant for lines 206-213
        #existing_files_list_str = "\n  - ".join(str(file.resolve()) for file in existing_files_list)

        raise FileExistsError(f"Save aborted! File was detected and overwrite protections are enabled: \n"
                              f"- {existing_files_list_str}\n"
                              "Either change overwrite protection in _base.yaml from io.allow_overwrite: true or select a different name for save"
        )




    with open(model_file_path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(encoder_file_path, "wb") as f:
        pickle.dump(encoder, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(label_binarizer_file_path, "wb") as f:
        pickle.dump(label_binarizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(metrics_file_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(parameters_file_path, "w", encoding="utf-8") as f:
        json.dump(parameters, f, indent=2)

    return {
        "model_path": model_file_path,
        "encoder_path": encoder_file_path,
        "label_binarizer_path": label_binarizer_file_path,
        "metrics_path": metrics_file_path,
        "parameters_path": parameters_file_path,
        "save_dir": save_dir,
    }


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def load_model_full(model_name, cfg):
    """ Loads pickle file from `model_name` and returns it."""

    io_cfg = cfg.get("io", {})

    model_dir = MODEL_DIR
    model_subdir = io_cfg.get("model_subdir")

    model_subdir_path = Path(model_dir) / model_subdir

    if not model_subdir_path.is_dir() or not model_subdir_path.exists():
        raise FileNotFoundError(f"Model subdir not found: {model_subdir_path}")


    model_path = model_subdir_path / f"{model_name}_model.pkl"
    encoder_path = model_subdir_path / f"{model_name}_encoder.pkl"
    label_binarizer_path = model_subdir_path / f"{model_name}_label_binarizer.pkl"

    paths = [model_path, encoder_path, label_binarizer_path]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing Component: {path}")
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.load(f)

    logger.info(type(model))
    logger.info(type(encoder))
    logger.info(type(label_binarizer))

    logger.info(hasattr(model, "fit"))  # should be True
    logger.info(hasattr(model, "predict"))  # should be True
    logger.info(hasattr(model, "n_features_in_"))  # confirms it was trained/fitted
    logger.info(hasattr(encoder, "categories_"))  # confirms fitted encoder
    logger.info(hasattr(label_binarizer, "classes_"))  # confirms fitted binarizer


    return model, encoder, label_binarizer


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)




#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, label_binarizer, model
):
    """ Computes the model metrics on a slice of the data specified by a column name and

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features: list
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    model : ???
        Model used for the task.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float

    """


    df_slice = data[data[column_name] == slice_value]


    X_slice, y_slice, _, _ = process_data(
        df_slice,
        categorical_features = categorical_features,
        label = label,
        training = False,
        encoder = encoder,
        lb = label_binarizer
    )


    #preds = None # your code here to get prediction on X_slice using the inference function
    slice_predictions = inference(model, X_slice)

    #precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    slice_precision, slice_recall, slice_fbeta = compute_model_metrics(y_slice, slice_predictions)


    #return precision, recall, fbeta
    return slice_precision, slice_recall, slice_fbeta 


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 





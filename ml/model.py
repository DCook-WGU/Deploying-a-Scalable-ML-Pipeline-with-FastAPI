import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data
# TODO: add necessary import

from sklearn.ensemble import RandomForestClassifier

from typing import Any, Dict, Optional
import importlib

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def _filter_params_for_cls(cls, params: Dict[str, Any]) -> Dict[str, Any]:

    try:
        inst = cls()  
        valid = set(getattr(inst, "get_params")().keys())
        return {key: value for key, value in params.items() if key in valid}
    except Exception:
        return params  

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def _build_estimator(cfg: Dict[str, Any] = None, class_path: str, parameters: Dict[str, Any] = None, default_random_state: int = 42):

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
    # TODO: implement the function
    #pass

    model = _build_estimator(cfg)
    model.fit(X_train, Y_train)

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
    # TODO: implement the function
    #pass

    if proba:
        probs = model.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int), probs
    else:
        return model.predict(X)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def save_model(model, encoder, lb, cfg, metrics, parameters, model_dir, model_name):
    """ Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    # TODO: implement the function
    #pass

    #suffixes = _add_suffixes(cfg)    
    
    #with open(, 'wb') as file:
    #    pickle.dump(model, file)



    '''
    if not io_cfg.get("allow_overwrite", True):
        expected = [path / name for name in io_cfg["file_name_suffixes"].values()]
        if any(path.exists() for path in expected):
            raise FileExistsError(f"Artifacts already exist under {path}")
    '''


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def load_model(path):
    """ Loads pickle file from `path` and returns it."""
    # TODO: implement the function
    pass



#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
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
    # TODO: implement the function
    X_slice, y_slice, _, _ = process_data(
        # your code here
        # for input data, use data in column given as "column_name", with the slice_value 
        # use training = False
    )
    preds = None # your code here to get prediction on X_slice using the inference function
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 





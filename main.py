import os
from functools import lru_cache
from typing import Dict, Tuple

import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model


from ml.paths import MODEL_DIR, DATA_DIR, CONFIGS_DIR, APP_ROOT


# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


def discover_models(MODEL_DIR):

    found_models = {}

    if not MODEL_DIR.exists():
        return found_models
    
    for model_subdir in MODEL_DIR.iterdir():

        if not model_subdir.is_dir():
            continue

        pickles = list(model_subdir.glob("*.pkl"))

        if not pickles:
            continue
        
        base_filenames = {p.stem.replace("_encoder", "").replace("_model","").replace("_label_binarizer","") for p in pickles}

        for base_filename in base_filenames:
            encoder_file = model_subdir / f"{base_filename}_encoder.pkl"
            model_file = model_subdir / f"{base_filename}_model.pkl"
            label_binarizer = model_subdir / f"{base_filename}_label_binarizer.pkl"

            if encoder_file.exists() and model_file.exists() and label_binarizer.exists():
                found_models[base_filename] = {"model_subdir": model_subdir.name, "base_filename": base_filename}

    return found_models


available_models = discover_models(MODEL_DIR)
default_model = "random_forest" if "random_forest" in available_models else next(iter(available_models))

@lru_cache(maxsize=16)
def get_model_encoder_label_binarizer(model_key):

    if model_key not in available_models:
        raise KeyError(f"Model, {model_key}, not available. Available models are: {list(available_models)}")


    selected_model = available_models[model_key]    

    model_base_path = MODEL_DIR
    model_subdir_path = model_base_path / selected_model["model_subdir"]
    model_filename = selected_model["base_filename"]

    model_filepath = model_subdir_path / f"{model_filename}_model.pkl"
    encoder_filepath = model_subdir_path / f"{model_filename}_encoder.pkl"
    label_binarizer_filepath = model_subdir_path / f"{model_filename}_label_binarizer.pkl"


    if not model_filepath.exists() or not encoder_filepath.exists() or not label_binarizer_filepath.exists():
        raise FileNotFoundError(f"Missing model components in {model_subdir_path}")

    encoder = load_model(encoder_filepath)
    model = load_model(model_filepath)
    label_binarizer = load_model(label_binarizer_filepath)

    return encoder, model, label_binarizer, model_subdir_path

#path = None # TODO: enter the path for the saved encoder 
#encoder = load_model(path)

#path = None # TODO: enter the path for the saved model 
#model = load_model(path)



# TODO: create a RESTful API using FastAPI
#app = None # your code here
app = FastAPI()

# Adding startup to print a list of available models
@app.on_event("startup")
async def startup_event():
    print(f"Discovered models: {list(available_models.keys())}")


# TODO: create a GET on the root giving a welcome message
@app.get("/")
async def get_root():
    """ Say hello!"""
    # your code here
    #pass

    return {
        "message": "Income classifier is up. POST to /data/?model=<key> with a record to get a prediction.",
        "models_available": list(available_models.keys()),
        "default_model": default_model,
        "model_dir": str(MODEL_DIR),
    }


# TODO: create a POST on a different path that does model inference
@app.post("/data/")
async def post_inference(data: Data,
                         model: str = Query(default_model, description=f"One of: {list(available_models.keys())}")):

    try:
        encoder, model, label_binarizer, model_subdir_path = get_model_encoder_label_binarizer(model)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # DO NOT MODIFY: turn the Pydantic model into a dict.
    data_dict = data.dict()
    # DO NOT MODIFY: clean up the dict to turn it into a Pandas DataFrame.
    # The data has names with hyphens and Python does not allow those as variable names.
    # Here it uses the functionality of FastAPI/Pydantic/etc to deal with this.
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)

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
    data_processed, _, _, _ = process_data(
        # your code here
        # use data as data input
        # use training = False
        # do not need to pass lb as input

        data,
        categorical_features=cat_features,
        label = None,
        training = False, 
        encoder = encoder,
        #lb = label_binarizer
    )
    #_inference = None # your code here to predict the result using data_processed
    _inference = inference(model, data_processed)
    return {"result": apply_label(_inference)}

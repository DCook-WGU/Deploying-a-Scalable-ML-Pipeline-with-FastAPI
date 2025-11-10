# Model Card — Census Income Classifier 

## Overview
- **Project**: Deploying a Scalable ML Pipeline with FastAPI  
- **Task**: Binary classification — predict if annual income is `>50K` or `<=50K`.  
- **Primary model**: `sklearn.ensemble.RandomForestClassifier` (default fallback).  
- **Owner/Maintainer**: Danty Dwayne Cook / DCook-WGU>  
- **Last updated**: 2025-11-10  
- **Version**: v1.0.0

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

## Intended Use
- **Intended users**: Data/ML engineers evaluating deployment pipelines.
- **Intended domain**: Educational / demonstration; not for production decisions.
- **Primary use-case**: Show end-to-end pipeline (data → train → eval → serve).

**Out of scope / Misuse**
- Using predictions for real financial/employment decisions.
- Inferring sensitive attributes or making adverse decisions about individuals.

## Data
- **Source**: UCI Adult/Census dataset (via Udacity starter).  
- **Features** (examples): age, workclass, education, marital-status, occupation, relationship, race, sex, hours-per-week, native-country, etc.  
- **Target**: income (`>50K` vs `<=50K`).# Training Data

**Preprocessing**
- Categorical features one-hot encoded.
- Numerical features kept as-is (optional scaling not required for trees).
- Data split into a 90-10 ratio. 90% is given to the Stratified K-Folds Cross Validation split. 10% is held to be used for feature slicing performance calculations. 

**Known data limitations**
- U.S.-centric; imbalanced target; contains sensitive attributes (sex, race).
- Historical, may reflect societal bias.

## Training - Default Configuration
- **Algorithm**: RandomForestClassifier  (This runs by default)
- **Typical params**:  
  - `n_estimators: 100`  
  - `criterion: "gini"`
  - `max_depth: null`
  - `min_samples_split: 2`
  - `min_samples_leaf: 1`
  - `random_state: 42`
  - `min_weight_fraction_leaf: 0.0`
  - `max_features: "sqrt"`
  - `max_leaf_nodes: null`
  - `min_impurity_decrease: 0.0`
  - `bootstrap: true`
  - `oob_score: false`
  - `# n_jobs: null`
  - `# random_state: null`
  - `verbose: 0`
  - `warm_start: false`
  - `class_weight: null`
  - `ccp_alpha: 0.0`
  - `max_samples: null`
- **Infrastructure**: CPU; scikit-learn.

---

**CLI Commands**: In command, you can pass in a --config arguement with the location of the configratuion file. Unless input otherwise, the pipeline will train and predict using the Random Forest Classifier by default. However, if you call the 

```
python train_model.py --config configs/gaussian_nb.yaml
```

The pipeline will load the configuration file and using the configuration file, the system will import the correct modules/libraries, load the configuration file prefilled with the basic configuration parameters for the model, and then perform the training. Once the training has been complete, the pipeline will save the model file as a pickel file and store it in the models/model_name sub-directory. The pipeline will also export the encoder, label binarizer, a json file of the metrics obtained from the run, a json file containing all the parameters used, and finally the slice performance training output will be saved into a text file. 

When using the uvicorn, you can select the model to be used via the ...:8000/docs/ sub domain interfact or you can initialate a get/post to the application by adding /data/?model={model_name} to the url, replace {model_name} with the model desired. 

```
http://127.0.0.1:8000/data/?model=kn_neighbors

```

The web service will automatically display all available models that have been trained and stored inside the project. 

---


## Evaluation
- **Metrics**: F1 Score/FBeta Score, precision, recall
- **Baseline**: Majority class or simple logistic regression.  
- **Reporting**: Macro/weighted F1; classwise precision/recall to show imbalance impact.

--- 

## Metrics
- **Validation F1**: 0.19999291433430171  
- **Precision/Recall**: 1.0 / 0.9998582967266544
- **Notes**: Performance sensitive to categorical handling and class imbalance.

--- 

## Ethical Considerations
- **Sensitive attributes** present; risk of encoding bias.  
- **Fairness**: Report group metrics (e.g., by sex, race) when possible.  
- **Mitigations**:  
  - Avoid using the model for real-world decisions.  
  - Audit disparate impact; consider rebalancing or post-processing if required.


## Risks & Limitations
- Not calibrated; probability thresholds may not be meaningful.
- Features may be noisy/missing in other sources.

---

## Testing & Monitoring
- **Unit tests**: data availability, directory existance, read/write checks (see `\ml_test.py`).  
- **Operational checks** (suggested): input schema validators, null/type guards.

---

## Versioning & Reproducibility
- **Model & Components**: model, encoder, label_binarizer, parameters, metrics are all saved together in the models subdirect, `\models\random_forest`  
- **Tracking**: DVC for data & artifacts; Git for code; CI via GitHub Actions.
- **System**: Docker, WSL, Conda, Python.

---

## Caveats and Recommendations
- Further development of pytests.
- Artifact Source handling would be good to have. Meaning having the machine learning model, the encoder, and the data files stored on a remote source with the versioning and tagging functionality enabled. This way when the pipeline is served to whatever machine is running it, the model, encoder, and data files for the default configuration can be auto-downloaded and run with those. By using a system similar to weights and baises artifact tracking and sourcing, we could ensure that the latest is always given and any deviations or updates can be uploaded seperately. 

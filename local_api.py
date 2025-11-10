import requests
import argparse

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Train a model using the provided YAML configuration file."
    )

    # Configs
    parser.add_argument(
        "-m",
        "--model",
        required=False,
        help="Selected Model",
    )

    return parser.parse_args()


def main():

    # Parse Arguements
    args = parse_args()

    # Load Config via arguments
    if args.model and args.model.strip():
        model_name = args.model
        logger.info(f"Model Found: {model_name}")
    else:
        model_name = "random_forest"
        logger.info("No model specified, using default - Random Forest Classifier")

    base_url = "http://127.0.0.1:8000"

    r = requests.get(base_url)

    logger.info(f"Status Code: {r.status_code}")

    if r.status_code == 200:
        data = r.json()
        logger.info(f"Result: {data.get('message')}")
    else:
        logger.info("Get Request failed.")

    data = {
        "age": 37,
        "workclass": "Private",
        "fnlgt": 178356,
        "education": "HS-grad",
        "education-num": 10,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }

    r = requests.post(f"{base_url}/data/?model={model_name}", json=data)

    logger.info(f"Status Code: {r.status_code}")

    if r.status_code == 200:
        result = r.json()
        logger.info(f"Result: {result.get('result')}")
    else:
        logger.info("Post Request failed.")


if __name__ == "__main__":
    main()

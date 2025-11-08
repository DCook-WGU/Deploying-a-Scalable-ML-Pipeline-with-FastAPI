import json

import requests

import argparse


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
        print(f"Model Found: {model_name}")
    else:
        model_name = "random_forest"
        print(f"No model specified, using default - Random Forest Classifier")

    base_url = "http://127.0.0.1:8000"

    # TODO: send a GET using the URL http://127.0.0.1:8000
    #r = None # Your code here
    r = requests.get(base_url)

    # TODO: print the status code
    # print()
    print(f"Status Code: {r.status_code}")

    # TODO: print the welcome message
    # print()

    if r.status_code == 200:
        data = r.json()
        print(f"Result: {data.get('message')}")
    else:
        print("Get Request failed.")


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

    # TODO: send a POST using the data above
    #r = None # Your code here
    r = requests.post(f"{base_url}/data/?model={model_name}", json=data)


    # TODO: print the status code
    # print()

    print(f"Status Code: {r.status_code}")

    # TODO: print the result
    # print()

    if r.status_code == 200:
        result = r.json()
        print(f"Result: {result.get('result')}")
    else:
        print("Post Request failed.")

if __name__ == "__main__":
    main()
import numpy as np
import onnx
import os
import glob

import onnxruntime as rt
from onnx import numpy_helper
from onnx.mapping import TENSOR_TYPE_MAP
import subprocess
from carbontracker.tracker import CarbonTracker

import tempfile
import argparse

import yaml
import torch


def fetch_model_from_name(folder: str, model_name: str):
    os.chdir("./models")
    if not os.path.exists(f"{model_name}.onnx"):
        sub = subprocess.run(
            [
                f"wget https://github.com/onnx/models/raw/main/{folder}/{model_name}.onnx",
            ],
            shell=True,
            check=True,
            capture_output=True,
        )
        sub2 = subprocess.run(
            [
                f"wget https://github.com/onnx/models/raw/main/{folder}/turnkey_stats.yaml -o {model_name}.yaml",
            ],
            shell=True,
            check=True,
            capture_output=True,
        )
        print("Fetched model")
    else:
        print("Already downloaded model")
    with open(os.path.abspath(f"./{model_name}.onnx"), "rb") as stream:
        model = onnx.load(stream)
    os.chdir("..")
    return model


def generate_test_data(model: onnx.ModelProto, n=None, batch_dim=None):
    assert len(model.graph.input) == 1
    elem_type = TENSOR_TYPE_MAP[model.graph.input[0].type.tensor_type.elem_type]
    shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    if n is not None and batch_dim is not None:
        shape[batch_dim] = n
    generator = np.random.default_rng()
    return generator.random(size=shape, dtype=elem_type.np_dtype)


def inference(model_name, input_data, tracker):
    session = rt.InferenceSession(
        os.path.join(os.getcwd(), "models", f"{model_name}.onnx")
    )
    # Get the input name for the ONNX model
    input_name = session.get_inputs()[0].name

    # Get the output name for the ONNX model
    output_name = session.get_outputs()[0].name

    # Run the model with the input data
    tracker.epoch_start()
    results = [session.run([output_name], {input_name: x}) for x in input_data]
    tracker.epoch_end()

    # return results


parser = argparse.ArgumentParser(prog="ONNX CarbonTracker")
parser.add_argument("repo_folder")
parser.add_argument("model_name")
parser.add_argument(
    "--api_key", help="API key for electricitymaps. Example: --api_key=123abc"
)
if __name__ == "__main__":
    args = parser.parse_args()
    model = fetch_model_from_name(args.repo_folder, args.model_name)
    test_data = [generate_test_data(model) for i in range(100)]
    api_keys = None
    if args.api_key is not None:
        api_keys = {"electricitymaps": args.api_key}
    tracker = CarbonTracker(epochs=1, api_keys=api_keys, verbose=0)
    inference(args.model_name, test_data, tracker)
    # print(test_data)

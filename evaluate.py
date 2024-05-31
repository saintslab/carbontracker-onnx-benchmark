import numpy as np
import onnx
import glob

import onnxruntime as rt
from onnx import numpy_helper
from onnx.mapping import TENSOR_TYPE_MAP
from carbontracker.tracker import CarbonTracker

import tempfile
import argparse

import yaml
import torch

from model_zoo import fetch_model_from_name
from model import generate_test_data, inference
import os

parser = argparse.ArgumentParser(prog="ONNX CarbonTracker")
parser.add_argument(
    "model", help="Model to evaluate. Can either be a file name or a URL to its Github"
)
parser.add_argument(
    "--api_key", help="API key for electricitymaps. Example: --api_key=123abc"
)
parser.add_argument(
    "-n", help="Amount of datapoints to test inference over", default=100, type=int
)

def parse_argument(arg):
    if isinstance(args.model, str):
        if isinstance(args.model, str) and 'github' in args.model:
            suffix = args.model.split('github.com/onnx/models')[1].split('main/')[1].split('/')
            return '/'.join(suffix[:-1]), suffix[-1]
    else:
        raise Exception('model has to be string (either filename or url to model zoo)')

if __name__ == "__main__":
    args = parser.parse_args()
    n = args.n
    folder, model_name = parse_argument(args.model)
    model = fetch_model_from_name(folder, model_name)
    test_data = [generate_test_data(model) for i in range(n)]
    api_keys = None
    if args.api_key is not None:
        api_keys = {"electricitymaps": args.api_key}
    tracker = CarbonTracker(epochs=1, monitor_epochs=1000, api_keys=api_keys, verbose=0, ignore_errors=False)
    energy, emissions = inference(model_name, test_data, tracker)
    print(f'Total used energy: {sum(energy)} kWh')
    print(f'Energy per inference: {sum(energy)/n} kWh')
    print(f'Total emissions produced: {sum(emissions)} gCO2eq')
    print(f'Emissions produced per inference: {sum(emissions)/n} gCO2eq')

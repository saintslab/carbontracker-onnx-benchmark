import argparse
import os
import onnx

import pandas as pd
from tqdm import tqdm
from carbontracker.tracker import CarbonTracker
from onnx_opcounter import calculate_params

from model import generate_test_data, inference
from model_zoo import fetch_model_from_name
from blacklist import blacklist

parser = argparse.ArgumentParser(prog="ONNX CarbonTracker benchmarker")
parser.add_argument("folder", help="Folder of models to evaluate")
parser.add_argument(
    "-o",
    "--output",
    help="File to output to (recommend .csv)",
    default="results/benchmarks.csv",
)
parser.add_argument(
    "-n",
    "--data_size",
    help="Amount of datapoints to do inference on",
    type=int,
    default=100,
)
parser.add_argument(
    "--api_key", help="API key for electricitymaps. Example: --api_key=123abc"
)


# Benchmark entire model zoos and add to .csv for later data processing
def get_models(path):
    folder = os.path.abspath(path)
    if not os.path.isdir(folder):
        raise Exception(f"{folder} is not a folder")
    models = []
    for dir_item in os.listdir(folder):
        dir_path = os.path.join(folder, dir_item)
        # print("dir item", dir_item)
        if os.path.isdir(dir_path):
            models += get_models(dir_path)
        elif ".onnx" in dir_item:
            models += [dir_path]

    return models


def get_models_already_evaluated(output_file, n):
    file = os.path.abspath(output_file)
    if os.path.exists(output_file):
        df = pd.read_csv(file)
        return df.loc[df["n"] == n, "model"]
    else:
        return pd.Series()


def write_results(output_path, model_path: str, n, energy, emissions, calculate_params):
    record = {
        "model_path": [model_path],
        "model": [model_path.split("/")[-1]],
        "n": [n],
        "total_energy": [energy],
        "total_emissions": [emissions],
        "total_params": [calculate_params],
    }
    df = pd.DataFrame.from_dict(record)
    mode = "a" if os.path.exists(output_path) else "x"
    df.to_csv(output_path, mode=mode, header=(not os.path.exists(output_path)))


if __name__ == "__main__":
    args = parser.parse_args()
    api_keys = None
    if args.api_key is not None:
        api_keys = {"electricitymaps": args.api_key}
    models = get_models(args.folder)
    already_evaluated = get_models_already_evaluated(args.output, args.data_size)
    with tqdm(total=len(models)) as pbar:
        pbar.update(len(already_evaluated))
        for model_path in models:
            model_folder = "/".join(model_path.split("/")[-3:-1])
            model_name = model_path.split("/")[-1]
            if (
                not (already_evaluated == model_name).any()
                and model_name not in blacklist
            ):
                model = fetch_model_from_name(model_folder, model_name)
                model.metadata_props
                test_data = [generate_test_data(model) for i in range(args.data_size)]
                tracker = CarbonTracker(
                    epochs=1,
                    monitor_epochs=1000,
                    api_keys=api_keys,
                    verbose=0,
                    ignore_errors=False,
                    epochs_before_pred=2,
                )
                energy, emissions = inference(model_name, test_data, tracker)
                write_results(
                    args.output,
                    model_path,
                    args.data_size,
                    sum(energy),
                    sum(emissions),
                    calculate_params(model),
                )
                pbar.update(1)

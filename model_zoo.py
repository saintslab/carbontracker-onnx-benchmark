import os
import subprocess
import onnx


def fetch_model_from_name(folder: str, model_name: str):
    os.chdir("./models")
    if not os.path.exists(f"{model_name}"):
        sub = subprocess.run(
            [
                f"wget https://github.com/onnx/models/raw/main/{folder}/{model_name}",
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
    with open(os.path.abspath(f"./{model_name}"), "rb") as stream:
        model = onnx.load(stream)
    os.chdir("..")
    return model

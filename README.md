# Carbontracker ONNX model evaluation utility
This tool is made to evaluate the resource consumption of inference of [ONNX](https://onnx.ai/) models.
It accepts both models as a file name and as links models in the [ONNX model zoo](https://github.com/onnx/models).

For accurate measurements, we recommend supplying an API key for [Electricitymaps](https://www.electricitymaps.com/), which is used for supplying carbon intensity information.

Usage (from ONNX model zoo):
~~~
python evaluate.py Computer_Vision/adv_inception_v3_Opset18_timm adv_inception_v3_Opset18 --api_key=123abc
~~~

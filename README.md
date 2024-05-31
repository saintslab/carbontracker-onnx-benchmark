# Carbontracker ONNX model evaluation utility
This tool is made to evaluate the resource consumption of inference of [ONNX](https://onnx.ai/) models.
It accepts both models as a file name and as links models in the [ONNX model zoo](https://github.com/onnx/models).

For accurate measurements, we recommend supplying an API key for [Electricitymaps](https://www.electricitymaps.com/), which is used for supplying carbon intensity information.

Example usage (with model from ONNX model zoo):
~~~
python evaluate.py https://github.com/onnx/models/blob/main/Computer_Vision/adv_inception_v3_Opset16_timm/adv_inception_v3_Opset16.onnx -n=100 --api_key=123abc
~~~
Response:
~~~
Total used energy: 3.5969332475905836e-05 kWh
Energy per inference: 3.5969332475905836e-07 kWh
Total emissions produced: 0.0027336692681688437 gCO2eq
Emissions produced per inference: 2.7336692681688437e-05 gCO2eq
~~~

It will automatically use and track connected CPUs (Intel or Apple Silicon only) and GPUs (Nvidia only).

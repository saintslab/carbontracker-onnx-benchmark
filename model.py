import numpy as np
import onnx

import onnxruntime as rt
from onnx.mapping import TENSOR_TYPE_MAP
from carbontracker.tracker import CarbonTracker

import os


def generate_test_data(model: onnx.ModelProto, n=None, batch_dim=None):
    assert len(model.graph.input) == 1
    elem_type = TENSOR_TYPE_MAP[model.graph.input[0].type.tensor_type.elem_type]
    shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    if n is not None and batch_dim is not None:
        shape[batch_dim] = n
    generator = np.random.default_rng()
    return generator.random(size=shape, dtype=elem_type.np_dtype)


def _inference(model, input_data, tracker):
    session = rt.InferenceSession(model)
    # Get the input name for the ONNX model
    input_name = session.get_inputs()[0].name

    # Get the output name for the ONNX model
    output_name = session.get_outputs()[0].name

    energy = None
    # Run the model with the input data
    tracker.epoch_start()
    results = [session.run([output_name], {input_name: x}) for x in input_data]
    tracker.epoch_end()
    energy = tracker.tracker.total_energy_per_epoch()
    emissions = tracker._co2eq(energy)
    tracker.stop()
    return energy, emissions


def inference(model_name, input_data, tracker: CarbonTracker):
    return _inference(
        os.path.join(os.getcwd(), "models", f"{model_name}"), input_data, tracker
    )

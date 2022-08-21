import cv2
import numpy as np
from mouse_pointer_controller.openvino_model import ModelDefinition, load_with_IECore, OpenVinoModel


def test_load_model_extract_output():
    """ testing model loading + inference """
    # Getting model bin and xml file
    model_path = "models/intel/face-detection-adas-0001/FP32/face-detection-adas-0001"

    net, input_layer = load_with_IECore(
        ModelDefinition.from_path(model_path), device_name="CPU", num_requests=1
    )
    assert input_layer == "data"
    expected_input_shape = net.input_info[input_layer].input_data.shape
    height, width = expected_input_shape[-2:]
    assert (height, width) == (384, 672)

    image = (
        cv2.resize(
            cv2.cvtColor(cv2.imread("tests/data/equipe-du-cameroun-de-beach-soccer.jpg"), cv2.COLOR_BGR2RGB),
            (width, height),
        )
        .transpose((2, 0, 1))
        .reshape(1, 3, height, width)
    )
    results = net.infer({input_layer: image})
    confidence = results["detection_out"][0][0][:, 2]

    assert sum(confidence > .95) == 11 #  complete soccer team
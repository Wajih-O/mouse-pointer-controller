import logging
from dataclasses import dataclass

from time import monotonic, time
from typing import Dict, Optional, Tuple

import cv2

import numpy as np

from openvino.inference_engine import IECore
from openvino.inference_engine.ie_api import ExecutableNetwork
from pydantic import BaseModel

LOGGER = logging.getLogger()

import os


class ModelDefinition(BaseModel):
    structure: str  # structure path
    weights: str  # model

    @classmethod
    def from_path(cls, path_prefix: str) -> "ModelDefinition":
        return ModelDefinition(
            structure=f"{path_prefix}.xml", weights=f"{path_prefix}.bin"
        )


def preprocess_image_input(image: np.ndarray, output_shape: np.ndarray) -> np.ndarray:
    """Preprocess image to fit expected input shape
    :param image: np.ndarray (height, width, depth)
    :param output_shape: (batch=1, depth, output_height, output_width)
    :return: (OpenVinoModel) inferable image as np.ndarray
    """
    height, width = output_shape[-2:]
    return (
        cv2.resize(image, (width, height))
        .transpose((2, 0, 1))
        .reshape(1, 3, height, width)
    )


def load_with_IECore(
    model_definition: ModelDefinition,
    ie=IECore(),
    device_name: str = "CPU",
    num_requests: int = 1,
) -> Tuple[ExecutableNetwork, str]:
    """
    load a model using  IECore
    :param device_name: the device name ex: "CPU"
    :param model_definition: model definition as an OpenVinoModel
    :param num_request: number of the requests (default = 1)
    :return: the tuple (network, input_name)
    """
    net = ie.read_network(
        model=model_definition.structure, weights=model_definition.weights
    )
    exec_net = ie.load_network(
        network=net, device_name=device_name, num_requests=num_requests
    )
    input_layer = next(
        iter(net.input_info)
    )  # to check if it is equivalent to 2021 next(iter(model.inputs))
    # print(net.input_info[input_layer].input_data.shape)
    return exec_net, input_layer


class OpenVinoModel(BaseModel):
    """OpenVino model"""

    model_name: str
    model_directory: str
    device: str = "CPU"
    network: Optional[ExecutableNetwork] = None
    input_layer_name: Optional[str] = None
    # extensions = None (if the model requires plugin) # todo clean-up
    loading_time: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def expected_input_shape(self) -> Optional[np.ndarray]:
        """if model is loaded return"""
        if self.input_layer_name and self.network:
            try:
                return self.network.input_info[self.input_layer_name].input_data.shape
            except ValueError:
                LOGGER.error("extracting loading input shape")
        else:
            LOGGER.warning("the model is not loaded yet!")
        return None

    @property
    def check(self):
        """Check the model is loaded (via the network expected class)"""
        return self.network.__class__.__name__ == "ExecutableNetwork"

    @property
    def loading_summary(self) -> Dict:
        return {"model_name": self.model_name, "loading_time": self.loading_time}

    def load_model(self):
        """Load the model"""
        start = monotonic()
        self.network, self.input_layer_name = load_with_IECore(
            ModelDefinition.from_path(
                os.path.join(self.model_directory, self.model_name)
            ),
            device_name=self.device,
            num_requests=1,
        )
        loading_time = monotonic() - start
        self.loading_time = loading_time

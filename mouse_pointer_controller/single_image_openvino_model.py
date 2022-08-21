"""
 A (simple) single image input model (OpenVino model wrapper)

 * @author Wajih Ouertani
 * @email wajih.ouertani@gmail.com

 """

import logging

from time import monotonic
from typing import Callable, List, Optional

import numpy as np

from mouse_pointer_controller.openvino_model import (
    OpenVinoModel,
    preprocess_image_input,
)
from mouse_pointer_controller.utils import ImageDimension

LOGGER = logging.getLogger()


def timing(to_time):
    """timing decorator"""

    def wrapper(*args, **kwargs):
        start = monotonic()
        res = to_time(*args, **kwargs)
        time = monotonic() - start
        args[0].prediction_time.append(time)
        return res

    return wrapper


class SingleImageOpenVinoModel(OpenVinoModel):
    """Single image input OpenVino model"""

    prediction_time: List[float] = []

    @property
    def image_dimension(self) -> Optional[np.ndarray]:
        return ImageDimension(*self.expected_input_shape[-2:][::-1])

    def preprocess_input(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image to fit expected input shape
        :param image: np.ndarray
        :return: inferable image as np.ndarray
        """
        if image.shape == self.expected_input_shape:
            return image.copy()
        return preprocess_image_input(image, self.expected_input_shape)

    @timing
    def predict(self, image):
        """
        Predict image
        """
        return self.network.infer({self.input_layer_name: image})

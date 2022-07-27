"""
 A (simple) single image input model (OpenVino model wrapper)

 * @author Wajih Ouertani
 * @email wajih.ouertani@gmail.com

 """

import logging
from typing import Optional

import cv2

import numpy as np

from mouse_pointer_controller.openvino_model import (
    OpenVinoModel,
    preprocess_image_input,
)
from mouse_pointer_controller.utils import ImageDimension

LOGGER = logging.getLogger()


class SingleImageOpenVinoModel(OpenVinoModel):
    """Single image input OpenVino model"""

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

    def predict(self, image):
        """
        Predict image
        """
        return self.network.infer({self.input_layer_name: image})

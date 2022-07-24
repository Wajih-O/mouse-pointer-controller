"""
LandmarksRegression class (OpenVino model wrapper)

 * @author Wajih Ouertani
 * @email wajih.ouertani@gmail.com
 """

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from mouse_pointer_controller.single_image_openvino_model import (
    SingleImageOpenVinoModel,
)
from mouse_pointer_controller.utils import Crop, RatioBoundingBox, RatioPoint
from pydantic import BaseModel


@dataclass
class EyeLandmark(RatioBoundingBox):
    """Eye landmark"""

    center: RatioPoint  # relative landmark center to the face image

    @classmethod
    def extract_eye_bbox(cls, landmark: RatioPoint, half_width=0.15, half_height=0.08):
        """Post process eye landmark (center) extracting a ratio bounding box surrounding it"""
        # refactor this first level extraction to use Crop class
        start, end = landmark.surround(half_width=half_width, half_height=half_height)
        return EyeLandmark(center=landmark, top_left=start, bottom_right=end)


class EyesLandmarks(BaseModel):
    left: EyeLandmark
    right: EyeLandmark


class LandmarksRegression(SingleImageOpenVinoModel):
    """Landmarks regression model class
    docs: https://docs.openvino.ai/latest/omz_models_model_landmarks_regression_retail_0009.html
    """

    model_name: str = "landmarks-regression-retail-0009"
    model_directory: str = "models/intel/landmarks-regression-retail-0009/FP32"

    def extract_landmarks(
        self,
        image,
    ) -> Dict:
        """Extract faces
        :param image: face image
        """
        if not self.check:
            # load model
            self.load_model()
            if not self.check:
                raise Exception("Could not load the model")

        to_infer = image.copy()
        if to_infer.shape != self.expected_input_shape:
            to_infer = self.preprocess_input(to_infer)

        landmarks = self.predict(to_infer)
        return landmarks

    def extract_eyes(self, image) -> Dict:
        """Extract eyes as landmarks from a given face image
        :param image: face image
        :return: left right
        """
        landmarks = self.extract_landmarks(image)
        eyes = np.split(list(landmarks.values())[0].flatten(), range(2, 10, 2))[
            :2
        ]  # as eyes are the first 2 landmarks

        right_eye, left_eye = map(
            lambda landmark: EyeLandmark.extract_eye_bbox(RatioPoint(*landmark)), eyes
        )
        return EyesLandmarks(left=left_eye, right=right_eye)

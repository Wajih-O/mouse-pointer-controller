"""
 Face detector class (OpenVino model wrapper)

 * @author Wajih Ouertani
 * @email wajih.ouertani@gmail.com
 """

import logging
from typing import List

import numpy as np

from mouse_pointer_controller.single_image_openvino_model import (
    SingleImageOpenVinoModel,
)
from mouse_pointer_controller.utils import RatioDetection, RatioPoint

LOGGER = logging.getLogger()


class FaceDetector(SingleImageOpenVinoModel):

    """Face Detection Model class"""

    model_name: str = "face-detection-adas-0001"
    model_directory: str = "models/intel/face-detection-adas-0001/FP32"

    def extract_faces(
        self,
        image: np.ndarray,
        detections=None,
        min_confidence: float = 0.95,
    ) -> List[RatioDetection]:
        """Extract faces
        :param image: model ready preprocessed image
        :param output_dimension: output dimension for each of the detected face
        :param detections: detection if set to None an inference is run to generate the detections
        :param min_confidence: minimum confidence to filter detections

        :return: faces crops and a visualization image (with bounding boxes) as a Tuple[List[np.ndarray], np.ndarray]
        """
        # Check if the model is loaded correctly!
        if detections is None:
            # check if the image is matching the expected dimension otherwise pre-process it
            to_infer = image.copy()
            if to_infer.shape != self.expected_input_shape:
                to_infer = self.preprocess_input(to_infer)
            # run inference
            detections = self.predict(to_infer)["detection_out"][0][0]
            # assume the batch size == 1 (we predict one image at once)
            image_ = to_infer[0].transpose(1, 2, 0)  # reshaped-back image

        # filter the detection to keep only high confidence detection (bounding boxes)
        filtered = detections[detections[:, 2] >= min_confidence]

        LOGGER.debug(
            "detections (at confidence > %.2f): %d", min_confidence, filtered.shape[0]
        )

        faces: List[RatioDetection] = []

        for _, _, confidence, start_x, start_y, end_x, end_y in filtered:
            faces.append(
                RatioDetection(
                    top_left=RatioPoint(start_x, start_y),
                    bottom_right=RatioPoint(end_x, end_y),
                    confidence=confidence,
                )
            )

        return faces

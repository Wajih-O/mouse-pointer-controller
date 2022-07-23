"""
 Face detector  class (OpenVino model wrapper)

 * @author Wajih Ouertani
 * @email wajih.ouertani@gmail.com
 """

import logging
import os
from typing import List, Optional, Tuple

import cv2

import numpy as np

from mouse_pointer_controller.single_image_openvino_model import (
    SingleImageOpenVinoModel,
)
from mouse_pointer_controller.utils import ImageDimension, RatioPoint

LOGGER = logging.getLogger()


class FaceDetector(SingleImageOpenVinoModel):

    """Face Detection Model class"""

    model_name: str = "face-detection-adas-0001"
    model_directory: str = "models/intel/face-detection-adas-0001/FP32"

    def extract_faces(
        self,
        image,
        output_dimension: Optional[ImageDimension] = None,
        detections=None,
        min_confidence: float = 0.95,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
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

        faces = []  # assuming
        image_dimension = self.image_dimension
        visu_image = image_.copy()
        for _, _, confidence, start_x, start_y, end_x, end_y in filtered:
            start = RatioPoint(start_x, start_y).project(image_dimension)
            end = RatioPoint(end_x, end_y).project(image_dimension)

            visu_image = cv2.rectangle(
                visu_image, start.as_array, end.as_array, color=(255, 0, 0), thickness=2
            )
            try:
                # original size
                LOGGER.debug("width: %d  height: %d", end.x - start.x, end.y - start.y)
                face_crop = image_[start.y : end.y, start.x : end.x]
                if output_dimension is not None:
                    face_crop = cv2.resize(face_crop, output_dimension.as_array)
                LOGGER.info("crop shape: %s", face_crop.shape)
                faces.append(face_crop)
            except Exception:
                LOGGER.exception("error while extracting faces cropping image")

        return faces, visu_image

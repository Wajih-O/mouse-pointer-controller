"""
HeadPoseEstimator class (OpenVino model wrapper)

 * @author Wajih Ouertani
 * @email wajih.ouertani@gmail.com
 """

from typing import Dict

import numpy as np
from mouse_pointer_controller.single_image_openvino_model import (
    SingleImageOpenVinoModel,
)


class HeadPoseEstimator(SingleImageOpenVinoModel):
    """Head pose estimation class"""

    model_name: str = "head-pose-estimation-adas-0001"
    model_directory: str = "models/intel/head-pose-estimation-adas-0001/FP32"

    def head_pose(
        self,
        face_image,
    ) -> Dict[str, np.ndarray]:
        """Extract faces
        :param face_image: face image
        """
        if not self.check:
            # load model
            self.load_model()
            if not self.check:
                raise Exception("Could not load the model")

        # check if the image is matching the expected dimension otherwise pre-process it
        to_infer = face_image.copy()
        if to_infer.shape != self.expected_input_shape:
            to_infer = self.preprocess_input(to_infer)

        return self.predict(to_infer)

"""
HeadPoseEstimator class (OpenVino model wrapper)

 * @author Wajih Ouertani
 * @email wajih.ouertani@gmail.com
 """

from dataclasses import dataclass
from operator import itemgetter
from typing import Dict

import numpy as np

from mouse_pointer_controller.single_image_openvino_model import (
    SingleImageOpenVinoModel,
)


@dataclass
class HeadPose:

    yaw: float
    pitch: float
    roll: float

    @classmethod
    def from_head_pose_estimator_output(cls, head_pose_dict: Dict[str, np.array]):
        """Build HeadPose from  a pose estimator output
           Example:
            {'angle_p_fc': array([[4.490898]], dtype=float32),
            'angle_r_fc': array([[-1.581182]], dtype=float32),
            'angle_y_fc': array([[-1.8917689]], dtype=float32)
        }

        """
        yaw, pitch, roll = itemgetter("angle_y_fc", "angle_p_fc", "angle_r_fc")(
            {key: value.flatten()[0] for key, value in head_pose_dict.items()}
        )
        return HeadPose(yaw=yaw, pitch=pitch, roll=roll)

    @property
    def as_array(self):
        return np.array([self.yaw, self.pitch, self.roll])


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

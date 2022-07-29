"""
GazeEstimation class (OpenVino model wrapper)

 * @author Wajih Ouertani
 * @email wajih.ouertani@gmail.com
 """

from dataclasses import dataclass
from operator import le

import numpy as np

from mouse_pointer_controller.head_pose_estimation import HeadPose, HeadPoseEstimator
from mouse_pointer_controller.landmarks_regression import (
    EyesLandmarks,
    LandmarksRegression,
)
from mouse_pointer_controller.openvino_model import (
    OpenVinoModel,
    preprocess_image_input,
)


@dataclass
class GazeEstimation:
    x: float
    y: float
    z: float

    @classmethod
    def from_array(cls, gaze_vector: np.ndarray):
        """Build a GazeEstimation object from a gaze_vector"""
        if gaze_vector.shape != (3,):
            raise ValueError("expect a 3d vector")
        x, y, z = gaze_vector
        return GazeEstimation(x=x, y=y, z=z)


@dataclass
class GazeEstimationResult:
    head_pose: HeadPose
    eyes_landmarks: EyesLandmarks
    gaze: GazeEstimation


class GazeEstimator(OpenVinoModel):
    """GazeEstimator model class
    docs: https://docs.openvino.ai/latest/omz_models_model_gaze_estimation_adas_0002.html
    """

    model_name: str = "gaze-estimation-adas-0002"
    model_directory: str = "models/intel/gaze-estimation-adas-0002/FP32"

    landmarks_regressor: LandmarksRegression
    head_pose_estimator: HeadPoseEstimator

    def infer(
        self,
        head_pose: HeadPose,
        left_eye_image: np.ndarray,
        right_eye_image: np.ndarray,
    ):
        """Infer gaze estimation from head pose and left, right eyes images"""

        return self.network.infer(
            {
                "head_pose_angles": head_pose.as_array,
                "left_eye_image": left_eye_image,
                "right_eye_image": right_eye_image,
            }
        )

    def estimate_gaze(self, face_image: np.ndarray):
        """Estimate gaze from a face image using the landmarks regression
        model for eye detection and the head pose estimator

        :param face_image: face image as np.ndarray"""

        if not self.check:
            # load model
            self.load_model()
            if not self.check:
                raise Exception("Could not load the model")

        head_pose = HeadPose.from_head_pose_estimator_output(
            self.head_pose_estimator.head_pose(face_image)
        )

        eyes_landmarks = self.landmarks_regressor.extract_eyes(face_image)

        left_eye_crop = preprocess_image_input(
            eyes_landmarks.left.crop(face_image),
            self.network.input_info.get("left_eye_image").input_data.shape,
        )
        right_eye_crop = preprocess_image_input(
            eyes_landmarks.right.crop(face_image),
            self.network.input_info.get("right_eye_image").input_data.shape,
        )

        gaze_vector = self.infer(
            head_pose=head_pose,
            left_eye_image=left_eye_crop,
            right_eye_image=right_eye_crop,
        )
        return gaze_vector

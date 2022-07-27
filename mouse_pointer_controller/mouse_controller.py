#!/usr/bin/env python
import logging
import os
from dataclasses import dataclass
from typing import List

import cv2

import numpy as np
import pyautogui
from commandr import command, Run

from input_feeder import InputFeeder

from video_utils import codec

from mouse_pointer_controller.face_detection import FaceDetector
from mouse_pointer_controller.gaze_estimation import GazeEstimator
from mouse_pointer_controller.head_pose_estimation import HeadPoseEstimator
from mouse_pointer_controller.landmarks_regression import LandmarksRegression
from mouse_pointer_controller.utils import (
    BoundingBox,
    ImageDimension,
    Point,
    RatioBoundingBox,
)


LOGGER = logging.getLogger("mouse-controller")


@dataclass
class PointerPrecision:
    """A helper mouse pointer precision class"""

    horizontal: int
    vertical: int

    @classmethod
    def from_ratio(cls, precision: float):
        screen_size = pyautogui.size()
        return PointerPrecision(
            horizontal=int(screen_size.width * precision),
            vertical=int(screen_size.height * precision),
        )


@dataclass
class MouseController:
    """A helper class to move the mouse relative to its current position"""

    precision: float = 0.2
    speed: float = 0.2

    @property
    def pointer_precision(self):
        return PointerPrecision.from_ratio(self.precision)

    def move(self, horizontal: float, vertical: float):
        screen_size = pyautogui.size()
        cursor_position = pyautogui.position()

        horizontal_move = horizontal * self.pointer_precision.horizontal

        if horizontal_move + cursor_position.x > screen_size.width:
            horizontal_move = screen_size.width - cursor_position.x
        if horizontal_move + cursor_position.x < 0:
            horizontal_move = -cursor_position.x

        # TODO: adjust vertical move

        pyautogui.moveRel(
            horizontal_move,
            -1 * vertical * self.pointer_precision.vertical,
            duration=self.speed,
        )

    def center(self):
        """Put the mouse cursor at the center of the screen"""


@dataclass
class HighPrecisionMouseController(MouseController):
    """a high precision slow MouseController"""

    precision: float = 0.01
    speed: float = 0.1


@command
def start(
    models_root_dir="./models/intel",
    input_type="video",
    input_file="./tests/data/demo.mp4",
    output_video_file="./output.mp4",
    gaze_estimation_data_output="./gaze_estimation_data.csv",
):
    """A main gaze based mouse pointer controller function:
    1 - Build and compose with the needed models
    2 - Extract face, eye landmark, head position -> then estimate gaze
    3 - Controls the mouse using the x,y from the gaze estimation and a MouseController + output and store the detection
    """

    feed: InputFeeder = InputFeeder(input_type=input_type, input_file=input_file)

    # Build/Prepare models
    landmarks_regressor = LandmarksRegression(
        model_directory=os.path.join(
            models_root_dir, "landmarks-regression-retail-0009/FP32"
        )
    )
    head_pose_estimator = HeadPoseEstimator(
        model_directory=os.path.join(
            models_root_dir, "head-pose-estimation-adas-0001/FP32"
        )
    )

    gaze_estimator = GazeEstimator(
        model_directory=os.path.join(models_root_dir, "gaze-estimation-adas-0002/FP32"),
        landmarks_regressor=landmarks_regressor,
        head_pose_estimator=head_pose_estimator,
    )

    face_detector = FaceDetector(
        model_directory=os.path.join(models_root_dir, "face-detection-adas-0001/FP32")
    )
    face_detector.load_model()

    feed.load_data()
    input_dimension = feed.dimension
    output_dimension = input_dimension.scale(0.3)
    print(tuple(face_detector.image_dimension.as_array))

    video_writer = cv2.VideoWriter(
        output_video_file,
        codec(),
        30,
        (output_dimension.width, output_dimension.height),
    )

    sample_size = None

    pointer_controller = HighPrecisionMouseController()
    # TODO: collect the gaze extraction into a Data-frame -> csv file
    gaze_data: List[np.ndarray] = []
    # TODO: remove the enumeration or better use it
    for _, frame in enumerate(feed.next_batch(sampling_rate=1, limit=sample_size)):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_detector.extract_faces(image)
        crops: List[RatioBoundingBox] = []
        if len(faces):
            selected_face = faces[0]
            crops.append(selected_face)
            face_crop = selected_face.crop(image)

            eyes_landmarks = landmarks_regressor.extract_eyes(face_crop)

            try:
                # Gaze estimation
                gaze_vector = gaze_estimator.estimate_gaze(face_crop)["gaze_vector"][0]
                gaze_data.append(gaze_vector)

                x, y, _ = gaze_vector
                pointer_controller.move(x, y)

                output_frame = image.copy()
                for eye_landmark in [eyes_landmarks.left, eyes_landmarks.right]:
                    bbox = BoundingBox(
                        Point(0, 0),
                        Point(input_dimension.width, input_dimension.height),
                    )
                    offset = ImageDimension.from_point(bbox.top_left)

                    for rbox in [selected_face, eye_landmark]:
                        bbox, offset = rbox.project_with_offset(bbox.dimension, offset)

                    output_frame = cv2.rectangle(
                        output_frame,
                        bbox.top_left.as_array,
                        bbox.bottom_right.as_array,
                        color=(0, 0, 255),
                        thickness=1,
                    )

                # draw the most confident face (as we expect one face in the application)
                output_frame = cv2.cvtColor(
                    cv2.resize(
                        selected_face.draw(output_frame),
                        (output_dimension.width, output_dimension.height),
                    ),
                    cv2.COLOR_RGB2BGR,
                )
                cv2.imshow(
                    "video mouse controller",
                    output_frame,
                )
                cv2.waitKey(1)
                video_writer.write(output_frame)

            except pyautogui.FailSafeException as fse:
                LOGGER.error("pyautogui.FailSafeException error!")

    video_writer.release()
    feed.close()


if __name__ == "__main__":
    Run()

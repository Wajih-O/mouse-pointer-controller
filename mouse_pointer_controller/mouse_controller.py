#!/usr/bin/env python
"""
 Main computer pointer (mouse) controller + models output visu./perf. data utils

 * @author Wajih Ouertani
 * @email wajih.ouertani@gmail.com

 """
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass

from typing import List

import cv2

import numpy as np
import pyautogui
from commandr import command, Run

from PIL import ImageGrab

from mouse_pointer_controller.face_detection import FaceDetector
from mouse_pointer_controller.gaze_estimation import GazeEstimationResult, GazeEstimator
from mouse_pointer_controller.head_pose_estimation import HeadPoseEstimator

from mouse_pointer_controller.input_feeder import InputFeeder
from mouse_pointer_controller.landmarks_regression import LandmarksRegression
from mouse_pointer_controller.openvino_model import OpenVinoModel
from mouse_pointer_controller.utils import (
    BoundingBox,
    ImageDimension,
    Point,
    RatioBoundingBox,
    RatioPoint,
)

from mouse_pointer_controller.video_utils import codec


LOGGER = logging.getLogger("mouse-controller")


@dataclass
class PointerPrecision:
    """A helper mouse pointer precision class"""

    horizontal: int
    vertical: int

    @classmethod
    def from_ratio(cls, precision: float):
        screen_size = pyautogui.size()
        hyp = np.sqrt(screen_size.width**2 + (screen_size.height) ** 2)
        # print(hyp)
        return PointerPrecision(
            horizontal=int(hyp * precision),
            vertical=int(hyp * precision),
        )


@dataclass
class MouseController:
    """A helper class to move the mouse relative to its current position"""

    precision: float = 0.10
    speed: float = 0.2
    screen_size = pyautogui.size()

    @property
    def pointer_precision(self):
        return PointerPrecision.from_ratio(self.precision)

    def gaze_move(self, horizontal: float, vertical: float, margin=300):
        cursor_position = pyautogui.position()

        horizontal_move = horizontal * self.pointer_precision.horizontal
        vertical_move = -1 * vertical * self.pointer_precision.vertical

        # Adjust horizontal move
        if horizontal_move + cursor_position.x > self.screen_size.width - margin:
            horizontal_move = self.screen_size.width - cursor_position.x - margin
        if horizontal_move + cursor_position.x < margin:
            horizontal_move = -cursor_position.x + margin

        # Adjust vertical move
        if vertical_move + cursor_position.y > self.screen_size.height - margin:
            vertical_move = self.screen_size.height - cursor_position.y - margin
        if vertical_move + cursor_position.y < margin:
            vertical_move = -cursor_position.y + margin

        pyautogui.moveRel(
            horizontal_move,
            vertical_move,
            duration=self.speed,
        )

    def move_to(self, x_position: int, y_position: int):
        """move to a pixel position"""
        # todo: ensure the position is within the Screen limits!
        pyautogui.moveTo(x_position, y_position)

    def center(self):
        """Put the mouse cursor at the center of the screen"""
        self.move_to(self.screen_size.width // 2, self.screen_size.height // 2)


@dataclass
class HighPrecisionMouseController(MouseController):
    """a high precision slow MouseController"""

    precision: float = 0.08
    speed: float = 0.01


def screen_crop_with_pointer(screen_crop: BoundingBox, marker_color=(255, 0, 0)):
    """Extract crop with cursor"""
    cursor_position = pyautogui.position()
    output_image = np.array(
        ImageGrab.grab(
            bbox=(
                screen_crop.top_left.x,
                screen_crop.top_left.y,
                screen_crop.bottom_right.x,
                screen_crop.bottom_right.y,
            )
        )
    )
    if (cursor_position.x > screen_crop.top_left.x) and (
        cursor_position.x < screen_crop.bottom_right.x
    ):
        if (cursor_position.y > screen_crop.top_left.y) and (
            cursor_position.y < screen_crop.bottom_right.y
        ):
            cv2.drawMarker(
                output_image,
                position=(
                    cursor_position.x - screen_crop.top_left.x,
                    cursor_position.y - screen_crop.top_left.y,
                ),
                color=marker_color,
                thickness=4,
            )

    return output_image


def ensure_output_directory(output_directory: str):
    """Ensure output directory exists (todo: try to create it)"""
    if os.path.exists(output_directory):
        if not os.path.isdir(output_directory):
            raise Exception(f"{output_directory} is not a directory")
    else:
        os.makedirs(output_directory)


@command
def start(
    models_root_dir="./models/intel",
    input_type="video",
    input_file="./tests/data/demo.mp4",
    model_precision="FP16",
    sample_size=None,
    output_directory="./output/",
):
    """A main gaze based mouse pointer controller function:
    1 - Build and compose with the needed models
    2 - Extract face, eye landmark, head position -> then estimate gaze
    3 - Controls the mouse using the x,y from the gaze estimation and a MouseController + output and store the detection


    :param models_root_dir : root directory for the (xml/bin) models
    :param input_type: input type, supports: "video" or "cam"
    :param input_file: data source when the type is set to video
    :param model_precision: model precision (default FP16)
    :param sample_size: to limit the frames number to consume from the input source
    :param output_directory: output directory for the generated artifacts control video capture and benchmarking data

    """

    ensure_output_directory(output_directory=output_directory)
    output_video_file = os.path.join(output_directory, "output.mp4")
    screen_capture_video_file = os.path.join(output_directory, "screen_capture.mp4")
    # Setup models:

    loading_summary = defaultdict(lambda x: [])
    face_detector = FaceDetector(
        model_directory=os.path.join(
            models_root_dir, "face-detection-adas-0001", model_precision
        )
    )
    face_detector.load_model()

    # Build Gaze estimator

    gaze_estimator = GazeEstimator(
        model_directory=os.path.join(
            models_root_dir, "gaze-estimation-adas-0002", model_precision
        ),
        landmarks_regressor=LandmarksRegression(
            model_directory=os.path.join(
                models_root_dir, "landmarks-regression-retail-0009", model_precision
            )
        ),
        head_pose_estimator=HeadPoseEstimator(
            model_directory=os.path.join(
                models_root_dir, "head-pose-estimation-adas-0001", model_precision
            )
        ),
    )

    gaze_estimator.landmarks_regressor.load_model()
    gaze_estimator.head_pose_estimator.load_model()
    gaze_estimator.load_model()

    models: List[OpenVinoModel] = [
        face_detector,
        gaze_estimator.head_pose_estimator,
        gaze_estimator.landmarks_regressor,
        gaze_estimator,
    ]

    # loading summary
    loading_summary = defaultdict(lambda: [])
    for model in models:
        loading_summary[model.loading_summary["model_name"]] = model.loading_summary[
            "loading_time"
        ]
    summary = {"precision": model_precision, "loading_time": loading_summary}
    with open(
        os.path.join(output_directory, f"loading_summary_{model_precision}.json"), "w"
    ) as loading_summary_output:
        json.dump(summary, loading_summary_output)

    print(summary)

    feed: InputFeeder = InputFeeder(input_type=input_type, input_file=input_file)
    feed.load_data()

    input_dimension = feed.dimension
    output_dimension = input_dimension.scale(0.3)

    video_writer = cv2.VideoWriter(
        output_video_file,
        codec(),
        30,
        (output_dimension.width, output_dimension.height),
    )

    # Desktop cropping (demo video output generation)
    screen_ratio_crop = RatioBoundingBox(
        top_left=RatioPoint(0.1, 0.1), bottom_right=RatioPoint(0.9, 0.9)
    )

    screen_dimension = ImageDimension.from_pyautogui_size(pyautogui.size())

    screen_crop: BoundingBox = screen_ratio_crop.project(screen_dimension)
    screen_crop_output_dimension = screen_crop.dimension.scale(0.5)
    print(screen_crop, screen_crop.dimension)

    screen_capture_writer = cv2.VideoWriter(
        screen_capture_video_file,
        codec(),
        30,
        screen_crop_output_dimension.as_array,  # screen_crop.dimension.as_array,
    )

    pointer_controller = HighPrecisionMouseController()
    pointer_controller.center()
    cv2.waitKey()

    main_window_name: str = "video mouse controller"
    # test_window_name: str = "screen crop"
    margin = 200
    cv2.namedWindow(main_window_name)
    cv2.moveWindow(main_window_name, margin + 100, margin + 100)
    pointer_controller.move_to(margin + 10, margin + 10)

    gaze_data: List[np.ndarray] = []  # collects the gaze estimation data

    # TODO: remove the enumeration or better use it
    for _, frame in enumerate(feed.next_batch(sampling_rate=1, limit=sample_size)):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_detector.extract_faces(image)
        crops: List[RatioBoundingBox] = []
        if len(faces):
            selected_face = faces[0]
            crops.append(selected_face)
            face_crop = selected_face.crop(image)

            try:
                # Gaze estimation
                gaze_estimation: GazeEstimationResult = gaze_estimator.estimate_gaze(
                    face_crop
                )
                gaze_data.append(gaze_estimation.gaze.as_array)

                x, y, _ = gaze_estimation.gaze.as_array
                pointer_controller.gaze_move(x, y, margin=margin)
                eyes_landmarks = gaze_estimation.eyes_landmarks
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

                # Draw the most confident face (as we expect one face in the application)
                output_frame = cv2.cvtColor(
                    cv2.resize(
                        selected_face.draw(output_frame),
                        (output_dimension.width, output_dimension.height),
                    ),
                    cv2.COLOR_RGB2BGR,
                )
                cv2.imshow(
                    main_window_name,
                    output_frame,
                )

                cv2.waitKey(1)

                # Screen capture  with cursor position drawing (as it is not captured with ImageGrab)
                desktop_crop = cv2.cvtColor(
                    cv2.resize(
                        screen_crop_with_pointer(screen_crop=screen_crop),
                        screen_crop_output_dimension.as_array,
                    ),
                    cv2.COLOR_RGB2BGR,
                )
                # cv2.imshow(test_window_name, desktop_crop)
                # cv2.waitKey(1)

                screen_capture_writer.write(desktop_crop)
                video_writer.write(output_frame)

            except pyautogui.FailSafeException as fse:
                LOGGER.error("pyautogui.FailSafeException error!")

    video_writer.release()
    feed.close()

    # saving perf. statistics/summary
    perf_stats = {"precision": model_precision, "average_prediction_time": {}}
    for model in models:
        if model.prediction_time:
            perf_stats["average_prediction_time"][model.model_name] = np.mean(
                model.prediction_time
            )

    print(perf_stats)
    with open(
        os.path.join(output_directory, f"perf_summary_{model_precision}.json"), "w"
    ) as perf_output:
        json.dump(perf_stats, perf_output)


if __name__ == "__main__":
    Run()

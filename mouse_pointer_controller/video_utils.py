from sys import platform

from typing import Callable, Iterable

import cv2
import numpy as np

Mat = "np.ndarray[int, np.dtype[np.generic]]"


def codec():
    if platform.startswith("linux"):
        return cv2.VideoWriter_fourcc(*"mp4v")
    else:
        if platform == "darwin":
            return cv2.VideoWriter_fourcc(*"MJPG")


def capture_stream(args, pipeline: Iterable[Callable[[Mat], Mat]]):

    # Capture
    cap = cv2.VideoCapture(args.i)  # make it a frame streamer as a part of the pipeline
    cap.open(args.i)

    # create a video writer for the output video
    video_writer = (
        cv2.VideoWriter("out.mp4", codec(), 30, (100, 100)) if not image_flag else None
    )

    # process frames until the video ends, or process is exited
    processed_frames = 0
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        for process in pipeline:
            frame = process(frame)
        processed_frames += 1

        if image_flag:
            cv2.imwrite("output_image.jpg", frame)
        else:
            video_writer.write(frame)

        # Break if escape key pressed
        if key_pressed == 27:
            break

    if not image_flag:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

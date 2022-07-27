from typing import Optional

import cv2

from mouse_pointer_controller.utils import ImageDimension


def consume_a_slot(slots: Optional[int]) -> Optional[int]:
    """Consume slots helper"""
    return slots - 1 if slots is not None else None


class InputFeeder:
    """
    A helper class to feed input from an image, webcam, or video.
    """

    def __init__(self, input_type, input_file=None):
        """
        input_type: str, The type of input, "video" for video file, "image" for image file,
                    or "cam" to use webcam feed.
        input_file: str, image or video file (ignored when input_type == "cam")
        """
        self.input_type = input_type
        if self.input_type in set({"video", "image"}):
            self.input_file = input_file
        self.capture = None

    def load_data(self):
        """Initialize/load the video (or) the image input"""
        if self.input_type == "video":
            self.capture = cv2.VideoCapture(self.input_file)
        elif self.input_type == "cam":
            self.capture = cv2.VideoCapture(0)
        else:
            self.capture = cv2.imread(self.input_file)

    def dimension(self) -> ImageDimension:
        return ImageDimension(
            width=self.capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            height=self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
        )

    def next_batch(self, sampling_rate: float = 0.1, limit: Optional[int] = None):
        """
        If input_type is 'image', then it returns self.capture (loaded image).
        otherwise it returns the next image from either a video file or webcam.
        """
        if self.input_type == "image":
            return self.capture

        slots = limit
        # subsample the original video only yield each 1/sampling_rate
        while slots is None or slots:
            flag, frame = self.capture.read()
            sample = 0
            while flag and sample < (int(1 / sampling_rate)):
                flag, frame = self.capture.read()
                sample += 1
            if not flag:
                break
            # update slots
            slots = consume_a_slot(slots)
            yield frame

    def close(self):
        """
        Closes the VideoCapture.
        """
        if not self.input_type == "image":
            self.capture.release()

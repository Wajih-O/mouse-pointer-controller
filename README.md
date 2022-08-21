# A Mouse/Computer pointer controller

## Project Set Up and Installation

### Installation

The project is pip installable using `pip (pip3)` (from the project root folder):

```bash
pip install .
```

### System dependencies (if not already installed)

```bash
sudo apt-get install python3-tk python3-dev
```

### Components, scripts and main program

The project provides `mouse_pointer_controller` as a library of utils (bounding-box, cropping, ... ) and OpenVino model wrappers:
`FaceDetector`, `HeadPoseEstimator`, `LandmarksRegression`, `GazeEstimator`

```python

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

```

It provides also `mouse_controller.py` as a runnable command/script (already in path after installation). To show the main script help `mouse_controller.py -h`
where `mouse_controller.py start -h` shows the main command `start` help:

```text

Documentation for command 'start':
----------------------------------------
A main gaze based mouse pointer controller function:
    1 - Build and compose with the needed models
    2 - Extract face, eye landmark, head position -> then estimate gaze
    3 - Controls the mouse using the x,y from the gaze estimation and a MouseController + output and store the detection

----------------------------------------

Usage: mouse_controller.py start [options]
Options without default values MUST be specified

Use: mouse_controller.py help [command]
  to see other commands available.

Options:
  -h, --help
  -m MODELS_ROOT_DIR, --models-root-dir=MODELS_ROOT_DIR
                        [default: "./models/intel"]
  -i INPUT_TYPE, --input-type=INPUT_TYPE
                        [default: "video"]
  -I INPUT_FILE, --input-file=INPUT_FILE
                        [default: "./tests/data/demo.mp4"]
  -M MODEL_PRECISION, --model-precision=MODEL_PRECISION
                        [default: "FP16"]
  -s SAMPLE_SIZE, --sample-size=SAMPLE_SIZE
                        [default: None]
  -o OUTPUT_DIRECTORY, --output-directory=OUTPUT_DIRECTORY
                        [default: "./output/"]

```

### Download models definition (XML, BIN)

The main script/program `mouse_controller` expects the BIN and XML for the needed models to be downloaded. The project also provides a convenient way to do that via `Makefile` running `./scripts/download_models.sh`:

```bash
make download-models
```

downloads all the needed models in `./models`

## Demo

After downloading the models, a demo using the default input video ("./tests/data/demo.mp4") (see command help `mouse_controller.py start`) is runnable with (this file could be replaced by any other relevant .mp4 using --input-file param)

```bash
mouse_controller.py start
```

for a specific model precision (ex: FP16-INT8):

```bash
mouse_controller.py start --model_precision="FP16-INT8"
```

<div>
<video controls width="500" src="output/screen_capture.mp4" muted="true">
</video>
</div>

## Benchmark

We report below model loading time, model inference time using (intel i9 9900k).

### Loading time

|           |   face-detection-adas-0001 |   head-pose-estimation-adas-0001 |   landmarks-regression-retail-0009 |   gaze-estimation-adas-0002 |
|:----------|---------------------------:|---------------------------------:|-----------------------------------:|----------------------------:|
| FP16-INT8 |                   0.191008 |                        0.0437531 |                          0.0428014 |                   0.0574666 |
| FP16      |                   0.09992  |                        0.0325476 |                          0.0189993 |                   0.0345011 |
| FP32      |                   0.120872 |                        0.0539329 |                          0.0225747 |                   0.0536712 |

### Perf./inference time

For each of the model we report the average over inference time over all demo video frames.

|           |   face-detection-adas-0001 |   head-pose-estimation-adas-0001 |   landmarks-regression-retail-0009 |
|:----------|---------------------------:|---------------------------------:|-----------------------------------:|
| FP16-INT8 |                  0.0084396 |                      0.000811234 |                        0.000348047 |
| FP16      |                  0.0115026 |                      0.000994762 |                        0.000351727 |
| FP32      |                  0.0106671 |                      0.00104288  |                        0.000367753 |

## Results

We remark a quasi-consistent increase in time perf. (a lower inference time) going from FP32 to FP16 to FP16-INT8 (INT8 quantized). While for the loading we remark that the quantized model takes the longest time.


## Multiple face detection

In case of a multiple faces detection we use the confidence to sort and select
the best available face detection. As it will not solve the problem of multiple confident face detection, for a pointer controller application in case of multiple detection the size of the face (bounding-box) is a good feature. Assuming the operator should be at a the closest distance from the screen expectedly will occupy the biggest bounding box.

## Generic pipeline inference

The pipeline inference is generic accepting both webcam and video

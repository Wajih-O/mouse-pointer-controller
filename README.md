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

### Components, scripts, and the main program

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

It provides also `mouse_controller.py` as a runnable command/script (already in the path after installation). To show the main script help `mouse_controller.py -h`
where `mouse_controller.py start -h` shows the main command `start` help:

```text

Documentation for command 'start':
----------------------------------------
A main gaze based mouse pointer controller function:
    1 - Build and compose with the needed models
    2 - Extract face, eye landmark, head position -> then estimate gaze
    3 - Controls the mouse using the x,y from the gaze estimation and a MouseController + output and store the detection


    :param models_root_dir : root directory for the (xml/bin) models
    :param input_type: input type, supports: "video" or "cam"
    :param input_file: data source when the type is set to video
    :param model_precision: model precision (default FP16)
    :param sample_size: to limit the frames number to consume from the input source
    :param output_directory: output directory for the generated artifacts control video capture and benchmarking data


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

A video capture of the controller in action is also attached in the output directory: `./output/screen_capture.mp4`

<div>
<video controls width="500" src="./output/screen_capture_h264.mp4"  muted="true">
</video>
</div>

## Benchmark

We report below model loading time, and model inference time using (intel i9 9900k).

### Loading time

|           |   face-detection-adas-0001 |   head-pose-estimation-adas-0001 |   landmarks-regression-retail-0009 |   gaze-estimation-adas-0002 |
|:----------|---------------------------:|---------------------------------:|-----------------------------------:|----------------------------:|
| FP16-INT8 |                  0.163184  |                        0.0426611 |                          0.0322367 |                   0.0559016 |
| FP16      |                  0.1232    |                        0.0338423 |                          0.0189576 |                   0.0412244 |
| FP32      |                  0.0888308 |                        0.0323096 |                          0.0187707 |                   0.0391196 |

### Perf./inference time

For each of the models, we report the average over inference time overall demo video frames.


|           |   face-detection-adas-0001 |   head-pose-estimation-adas-0001 |   landmarks-regression-retail-0009 |   gaze-estimation-adas-0002 |
|:----------|---------------------------:|---------------------------------:|-----------------------------------:|----------------------------:|
| FP16-INT8 |                  0.0079483 |                      0.000747055 |                        0.000336387 |                 0.000706338 |
| FP16      |                  0.0104724 |                      0.000996869 |                        0.000415242 |                 0.00107239  |
| FP32      |                  0.0108567 |                      0.0010213   |                        0.000357545 |                 0.00105316  |

## Results

We remark a quasi-consistent decrease in inference time (faster inference), going from FP32 to FP16 to FP16-INT8 (INT8 quantized). While for the loading we remark that the quantized model takes the longest time.


## Multiple face detection

In the case of multiple faces detection, we use the confidence to sort and select
the best available face detection. As it will not solve the problem of numerous confident face detection, for a pointer controller application in case of multiple detections the size of the face (bounding-box) is a good feature. Assuming the operator should be at the closest distance from the screen expectedly will occupy the biggest bounding box. Also for stability we can track the face bounding box keeping the closest to a first detection using the Jaccard Index.

## Generic pipeline inference

The pipeline inference is generic accepting both webcam and video

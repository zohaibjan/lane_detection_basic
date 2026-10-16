# Basic Lane Detection with OpenCV

A computer vision learning exercise created while studying self-driving car concepts. The script accepts an image or video file and a mode argument.

## Usage

Install NumPy, OpenCV and Matplotlib in your Python environment, then run from a desktop session with display support:

```bash
python main.py image.jpg 1
python main.py video.mp4 2
```

- `1`: image input.
- `2`: video input.

Place the input file in the working directory, or supply a suitable path.

## Scope

This is an educational image-processing experiment. It is not an autonomous-driving component or a validated road-safety system. Behaviour depends on the supplied imagery and the parameters in `main.py`.

[Profile and project directory](https://github.com/zohaibjan)

## Checks and limitations

The code handles frames without detected lanes, one-sided detections, vertical/horizontal segments and video end-of-file. Synthetic geometry checks run with `python -m unittest discover -s tests -v`; they do not validate camera footage or driving performance. The region-of-interest polygon remains tuned to the original image resolution and needs adjustment for other cameras.

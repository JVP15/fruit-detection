# Deep Fruit Vision

This program detects, classifies, identifies ripeness, and identifies maturity of fruit in an image, video, or camera stream.

## Installation

We incorporated Yolo-v5 (specifically release 6.2) for this project. You can download and install this project and its requirements with:

```
git clone https:
cd fruit-detection https://github.com/JVP15/fruit-detection.git
pip install -r requirements.txt
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```


## Datasets

You can automatically download most of the datasets by doing 
```
cd modules
python datasets.py
```

However, there are some datasets that need to be manually downloaded from Google Drive and renamed to the correct name. These are:
[Papaya Dataset](https://drive.google.com/file/d/1lg1gM_CtZGsGUrHeY8aZmrl3drG66kxX/view?usp=sharing)
[Ripeness Dataset](https://drive.google.com/drive/folders/1ZXSaUBMtR-nymOY-WaAz-iY3e-i2xoPf?usp=sharing) (rename to `ripeness_dataset.zip`)
[Defect Dataset](https://drive.google.com/drive/folders/1s7GP9iYfF5wgv1AmFbai2TqUHaUpMntG?usp=share_link) (rename to `defect_dataset.zip`)
[Ensemble Dataset](https://drive.google.com/file/d/1cjEaQInMgdh9cRaFhatao0FQP0Ar3dYV/view?usp=share_link)

## Running the Program

You can run Deep Fruit Vision by using: ```python run.py```

You can use the GUI version by adding the `--use-gui` argument like: ```python run.py --use-gui```





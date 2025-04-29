# Vision
## Description
This API provides tools to simplify training and running multiple AI models, as well as performing hardware stress tests and automatic performance logging.

### Features:
- __Train multiple AIs in a stack__

- __Run multiple AI models simultaneously with different settings__

- __Perform hardware stress tests__

- __Generate detailed performance logs__

- __Automate log processing and data visualization__

## How to use

1. Configure the AI models in the [configuration file](#Configuration-file)

2. Run in [live](#1-Live) or [train](#2-Train)

3. (optional) Run [data processing](#3-data-processing)



## How to run

```python3 main.py [mode] [-args]```

## modes
### 1. Live
run real-time detection<br>
```
python3 main.py live [-args]
```
<br>

__args:__

- ```-nv``` no video

- ```-cap``` capture objects (process image)

- ```-pl``` create [performance logs ](#Performance-Logs)

- ```-rtsp``` live from video of rtsp server

- ```-sf [frame_number]``` skip frames for better performance

- ```-rec [output_file_name]``` record

- ```-f [file_path]``` live from file
<br><br>

### 2. Train:
run train :)<br>
```
python3 main.py train
```

> [!NOTE]
> If segmentation IA, run [segmentation from boxes](#segmentation-from-boxes)

<br>

### 3. Data processing:
Generate plots from logs, separating _capture objects_ from _non capture objects_.<br>
```
python3 data_processing.py
```
Processed logs are saved in the ```logs_processed``` folder
<br>

## Performance Logs
Logs include time, ram usage, gpu usage, cpu usage, cycle time, active AIs and captre objects data.

> [!IMPORTANT]
> Logs are always saved on ```logs``` folder


## Configuration file

All AI settings are defined in the ```config.yaml``` file.

### Parameters:
- ```dataset:``` dataset path

- ```weights:``` weights path for detection

- ```confidence:``` confidence for detection

- ```label:``` class label

- ```epochs:``` train epochs

- ```device:``` cpu or gpu

- ```result folder name:``` name of folder for train results

- ```model:``` AI model

- ```detect:``` true or false, activate detection for this AI

- ```train:``` true or false, activate train for this AI

- ```segmentation:``` true or false, set true if segmentation AI 

## Segmentation from boxes
Create segmentation dataset from bounding boxes dataset

1. Configure paths in segmentation [config file](#segmentation-configuration-file)

2. In seg_from_boxes folder, run ```python3 seg.py```

### Segmentation configuration file

__parameters:__ 

- ```raw_path:``` path to folder containing raw images folder and labels folder

- ```save_path:``` path to save segmentation dataset

- ```save_path_prefix:``` create parent folder for save path, can be empty

- ```epochs:``` train epochs

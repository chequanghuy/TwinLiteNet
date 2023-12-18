# TwinLiteNet: An Efficient and Lightweight Model for Driveable Area and Lane Segmentation in Self-Driving Cars




## Requirement
See `requirements.txt` for additional dependencies and version requirements.

```setup
pip install -r requirements.txt
```


## Data Preparation

- Download the images from [images](https://bdd-data.berkeley.edu/).

- Download the annotations of drivable area segmentation from [segments](https://drive.google.com/file/d/1xy_DhUZRHR8yrZG3OwTQAHhYTnXn7URv/view?usp=sharing). 
- Download the annotations of lane line segmentation from [lane](https://drive.google.com/file/d/1lDNTPIQj_YLNZVkksKM25CvCHuquJ8AP/view?usp=sharing). 

```bash
/data
    bdd100k
        images
            train/
            val/
            test/
        segments
            train/
            val/
        lane
            train/
            val/
```
## Pipeline

<div align=center>
<img src='image\arch.png' width='600'>
</div>

## Train
```python
python3 main.py
```

## Test
```python
python3 val.py
```

## Inference

### Images
```python
python3 test_image.py
```

## Visualize
### Drive-able segmentation

<div align=center>
<img src='image\DA_vs.jpg' width='600'>
</div>
### Lane Detection

<div align=center>
<img src='image\LL_vs.jpg' width='600'>
</div>



## Acknowledgement
Our source code is inspired by:
- [ESPNet](https://github.com/sacmehta/ESPNet)
- [YOLOP](https://github.com/hustvl/YOLOP)



## Citation

If you find our paper and code useful for your research, please consider giving a star :star:   and citation :pencil: :

```BibTeX
@INPROCEEDINGS{10288646,
  author={Che, Quang-Huy and Nguyen, Dinh-Phuc and Pham, Minh-Quan and Lam, Duc-Khai},
  booktitle={2023 International Conference on Multimedia Analysis and Pattern Recognition (MAPR)}, 
  title={TwinLiteNet: An Efficient and Lightweight Model for Driveable Area and Lane Segmentation in Self-Driving Cars}, 
  year={2023},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/MAPR59823.2023.10288646}}
```

<div align="center">
  <img src="twin.png" width="30%">
</div>

# TwinLiteNetV2: A small stone can kill a giant
## 🚀 Coming soon! 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/twinlitenet-an-efficient-and-lightweight/lane-detection-on-bdd100k-val)](https://paperswithcode.com/sota/lane-detection-on-bdd100k-val?p=twinlitenet-an-efficient-and-lightweight)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/twinlitenet-an-efficient-and-lightweight/drivable-area-detection-on-bdd100k-val)](https://paperswithcode.com/sota/drivable-area-detection-on-bdd100k-val?p=twinlitenet-an-efficient-and-lightweight)

| Model | size<br><sup>(Height x Width) | Lane<br><sup>(Accuracy) | Lane<br><sup>(IOU) | Drivable Area<br><sup>(mIOU)  | params<br><sup>(M) | FLOPs<br><sup> (B) |
| ----- | ----------------------------- | ----------------------- | ------------------ | ----------------------------- | ----------------------------- | ----------------------------- |
| [TwinLiteNetV2-Nano]()| 384 x 640                   | 70.8 | 23.6              | 87.2                       | 0.03   | 0.485 |
| [TwinLiteNetV2-Small]()| 384 x 640                   | 75.9 | 28.7              | 90.4                      | 0.14   | 1.366 |
| [TwinLiteNetv2-Medium]()| 384 x 640                   | 79.3 | 32.6              | 92.3                     | 0.62   | 5.088 |
| [TwinLiteNetV2-Large]() | 384 x 640                   | 81.7 | 34.2              | 92.9                     | 2.78   | 21.526 |



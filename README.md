# TwinLiteNet: An Efficient and Lightweight Model for Driveable Area and Lane Segmentation in Self-Driving Cars
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/twinlitenet-an-efficient-and-lightweight/lane-detection-on-bdd100k-val)](https://paperswithcode.com/sota/lane-detection-on-bdd100k-val?p=twinlitenet-an-efficient-and-lightweight)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/twinlitenet-an-efficient-and-lightweight/drivable-area-detection-on-bdd100k-val)](https://paperswithcode.com/sota/drivable-area-detection-on-bdd100k-val?p=twinlitenet-an-efficient-and-lightweight)
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
@misc{che2023twinlitenet,
      title={TwinLiteNet: An Efficient and Lightweight Model for Driveable Area and Lane Segmentation in Self-Driving Cars}, 
      author={Quang Huy Che and Dinh Phuc Nguyen and Minh Quan Pham and Duc Khai Lam},
      year={2023},
      eprint={2307.10705},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

# Efficient-Semantic-Segmentation-for-Self-Driving-Cars

## Requirement
See `requirements.txt` for additional dependencies and version requirements.

```setup
pip install -r requirements.txt
```


## Data Preparation

### Download

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
# Pipeline

<div align=center>
<img src='image\arch.png' width='600'>
</div>

# Train
```python
python3 main.py
```

# Inference

## Images
```python
python3 test_image.py
```
## Video
```python
python3 test_video.py
```

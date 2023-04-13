# Efficient-Semantic-Segmentation-for-Self-Driving-Cars

# Prepare datasets

```bash
/data
    bdd100k
        train/
        val/
    bdd_lane_gt
        train/
        val/
    bdd_seg_gt
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
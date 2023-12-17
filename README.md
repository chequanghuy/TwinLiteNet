# TwinLiteNetV2: A small stone can kill a giant

## Requirement
See `requirements.txt` for additional dependencies and version requirements.

```setup
pip install -r requirements.txt
```

| Model | size<br><sup>(Height x Width) | Lane<br><sup>(Accuracy) | Lane<br><sup>(IOU) | Drivable Area<br><sup>(mIOU)  | params<br><sup>(M) | FLOPs<br><sup> (B) |
| ----- | ----------------------------- | ----------------------- | ------------------ | ----------------------------- | ----------------------------- | ----------------------------- |
| [TwinLiteNet-Nano]()| 384 x 640                   | 70.8 | 23.6              | 87.2                       | 0.03   | 0.485 |
| [TwinLiteNet-Small]()| 384 x 640                   | 75.9 | 28.7              | 90.4                      | 0.14   | 1.366 |
| [TwinLiteNet-Medium]()| 384 x 640                   | 79.3 | 32.6              | 92.3                     | 0.62   | 5.088 |
| [TwinLiteNet-Large]() | 384 x 640                   | 81.7 | 34.2              | 92.9                     | 2.78   | 21.526 |


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

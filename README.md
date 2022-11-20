# Project for Introduction to Auditory-visual Information System -- audio and video matching under noise interference

## records

report: https://www.overleaf.com/3393918221kzwhkjdscmfh

record: https://1drv.ms/x/s!Agcs68x5s4XGhngeyctknqAYWlr5?e=swsuEa

## usage

### data preparation

Extract the .zip file from , and move the feature file under `<extractor_name>`, like:

```
Train
├── afeat
│   └── vggish-quant
│   └── <other extractor name>
├── audio
├── vfeat
│   └── <other extractor name>
└── video
```

Then use `tools/generate_random_dataset.py` to split the `Train` dataset into `Train`(90%) and `Dev`(10%). Notice you should modify your absolute path in this script.

After split, the directory tree should like this:

```
data
└── Dataset
    ├── Dev
    │   ├── afeat
    │   │   └── vggish-quant
    │   └── vfeat
    │       └── resnet-101
    ├── Test
    │   ├── Clean
    │   │   ├── afeat
    │   │   │   └── vggish-quant
    │   │   ├── audio
    │   │   ├── vfeat
    │   │   │   └── resnet-101
    │   │   └── video
    │   └── Noise
    │       ├── afeat
    │       │   └── vggish-quant
    │       ├── audio
    │       ├── vfeat
    │       │   └── resnet-101
    │       └── video
    ├── Train
    │   ├── afeat
    │   │   └── vggish-quant
    │   ├── audio
    │   ├── vfeat
    │   │   └── resnet-101
    │   └── video
    └── Train_Part
        ├── afeat
        │   └── vggish-quant
        └── vfeat
            └── resnet-101
```

### matchnet training

Modify the `configs/train_config.yaml` , and then use `python train.py`

### matchnet training

Modify the `configs/test_config.yaml` , and then use `python test.py`
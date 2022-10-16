# TPSNet
A pytorch implementation for the ACM MM2022 paper "TPSNet: Reverse Thinking of Thin Plate Splines for Arbitrary Shape Scene Text Representation"

## Environment
This implementation is based on mmocr-0.2.1, so please refer to it for detailed requirements. Our code has been test with Pytorch-1.8.1 + cuda11.1
We recommend using [Anaconda](https://www.anaconda.com/) to manage environments. Run the following commands to install dependencies.
```
conda create -n tpsnet python=3.7 -y
conda activate tpsnet
 conda install pytorch=1.8 torchvision cudatoolkit=11.1 -c pytorch -c nvidia -c conda-forge
pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
pip install mmdet==2.14.0
git clone https://github.com/Wei-ucas/TPSNet
cd TPSNet
pip install -r requirements.txt
python setup.py build develop
cd mmocr/models/textend2end/utils/grid_sample_batch
python setup.py build develop
```

## Dataset
**Synthtext-150k**  [Source](https://github.com/aim-uofa/AdelaiDet/tree/master/configs/BAText)


**MLT17**[[source]](https://rrc.cvc.uab.es/?ch=8&com=introduction) 


**Total-Text** [[source]](https://github.com/cs-chan/Total-Text-Dataset). 


**CTW1500** [[source]](https://github.com/Yuliang-Liu/Curve-Text-Detector).

The prepared annotations can be download from [Google Drive](https://drive.google.com/drive/folders/1zUPTFXzJlBMlEu_hRjluutZGrEQ4Et1Y?usp=sharing), for synthtext-150k and MLT17 images, please download from the source above.

Please download and extract the above datasets into the `data` folder following the file structure below.
```
data
├─totaltext
│  │ totaltext_train.json
│  │ totaltext_test.json
│  └─imgs
│      ├─training
│      └─test
├─CTW1500
│  │ instances_training.json
│  │ instance_test.json
│  └─imgs
│      ├─training
│      └─test
├─mlt2017
│  │  train_polygon.json
│  └─MLT_train_images
├─syntext1
│  │  train_polygon.json
│  └─images
├─syntext2
│  │  train_polygon.json
│  └─images

```


## Train
### Pretrain
`CUDA_VISIBLE_DEVICES=0,1 ./tools/train.sh config/tpsnet/tpsnet_pretrain.py work_dirs/pretrain 2
`
### Finetune
`CUDA_VISIBLE_DEVICES=0,1 ./tools/train.sh config/tpsnet/tpsnet_totaltext.py work_dirs/totaltext 2 --load-from work_dirs/pretrain/latest.pth`

## Evaluation
`CUDA_VISIBLE_DEVICES=0 python tools/test.py config/tpsnet/tpsnet_totaltext.py work_dirs/totaltext/latest.pth --eval hmean-e2e`

The `hmean-e2e` evaluation code comes from [ABCNetV2](https://github.com/aim-uofa/AdelaiDet/tree/master/configs/BAText), see `mmocr/core/evaluation/evaluation_e2e` for details.


## Trained Model
Will be released soon.


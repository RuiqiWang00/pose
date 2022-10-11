# pose  
## Installation
>conda create -n pose python=3.9  
>conda activate pose  
>conda install pytorch torchvision cudatoolkit  
>pip install -r requirements.txt  

## prepare dataset   
./data  
  ├── annotations   
  ├── test2017  
  ├── train2017  
  └── val2017   
## download imagenet pretrained   
down from CID   
put it in  
>./model  
## Train Model
>python tools/train.py --cfg experiments/coco.yaml --gpus 0,1
## Evaluate Model
>python tools/valid.py --cfg experiments/coco-testdev.yaml --gpus 0,1 TEST.MODEL_FILE model/coco/checkpoint.pth.tar

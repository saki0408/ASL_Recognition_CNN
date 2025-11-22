# ASL Recognition using CNN

This real-time ASL recognition system uses transfer learning through MobileNet V2/EfficientNet B0 models.

## Key Features
- multiple CNN architectures: MobileNet V2, EfficientNet B0
- transfer learning with ImageNet pre-trained weights
- MediaPipe hand detection for real-time inference
- auto-resume training, CSV logging, checkpoint recovery
- temporal smoothening for prediction stability

### Clone repository
`git clone https://github.com/saki0408/ASL_Recognition_CNN.git`

`cd ASL_Recognition_CNN`

### Create virtual environment
`python -m venv asl_venv`
`asl_venv\Scripts\activate`

### Install dependencies
`pip install tensorflow>=2.10 keras opencv-python numpy pandas matplotlib mediapipe scikit-learn`

## Usage
### Train model
#### MobileNet V2
`python a.py --train --dataset_dir ./dataset --img_size 160 --batch_size 32 --model_arch mobilenetv2`
#### EfficientNet B0
`python a.py --train --dataset_dir ./dataset --img_size 224 --batch_size 32 --model_arch efficientnetb0`
#### With fine-tuning
`python a.py --train --dataset_dir ./dataset --img_size 224 --epochs 10 --fine_tune_epochs 4 --unfreeze_layers 50 --fine_tune_lr 1e-5`

### Real-time inference
`python webcam_infer.py`

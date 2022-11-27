# Depth-Aware Generative Adversarial Network for Talking Head Video Generation

## Installation
```bash
pip install -r requirements.txt

## Install the Face Alignment lib
cd face-alignment
pip install -r requirements.txt
python setup.py install
```


## Training
Please follow the instruction from https://github.com/AliaksandrSiarohin/video-preprocessing for VoxCeleb dataset.

To train a model on VoxCeleb dataset run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_addr="0.0.0.0" --master_port=12348 run.py --config config/vox-adv-256.yaml --name DaGAN --rgbd --batchsize 12 --kp_num 15 --generator DepthAwareGenerator
```

## Test
To test a model on Obama dataset download pre-trained checkpoints of face depth network and DaGAN: [OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/fhongac_connect_ust_hk/EjfeXuzwo3JMn7s0oOPN_q0B81P5Wgu_kbYJAh7uSAKS2w?e=KaQcPk).
```bash
python -u demo.py --config config/vox-adv-256.yaml --kp_num 15 --source_image config/source_image.jpg --driving_video config/driving_video.mp4 --relative --adapt_scale --generator DepthAwareGenerator
```

The code will create a ./result folder. Generated **lip-sync video** and **visualized keypoints** will be saved to this folder. The video is generated using source_image.jpg and driving_video.mp4 in ./config folder.

The code will also print **PSNR** and **SSIM** performance.

# WiSoDiffusionImage Super-Resolution via Iterative Refinement

[Paper](https://arxiv.org/pdf/2104.07636.pdf ) |  [Project](https://iterative-refinement.github.io/ )

## Brief

This is an unofficial implementation of WiSoDiffusion, an extension of [WiSoSuper] (https://arxiv.org/abs/2109.08770.pdf). The goal of this project is to apply a state-of the art diffusion-based model to the problem of super-resolving raw wind speed values. This has potential applications in renewable energy planning on local scales. This implementation uses SR3, the model proposed by "Image Super-Resolution via Iterative Refinement" ([paper] (https://arxiv.org/pdf/2104.07636.pdf)).

### Training Step

- [x] log / logger
- [x] metrics evaluation
- [x] multi-gpu support
- [x] resume training / pretrained model
- [x] validate alone script
- [x] Weights and Biases Logging

## Usage
### Environment
```python
pip install -r requirement.txt
```

### Pretrained Model

This paper is based on "Denoising Diffusion Probabilistic Models", and we build both DDPM/SR3 network structures, which use timesteps/gamma as model embedding inputs, respectively. In our experiments, the SR3 model can achieve better visual results with the same reverse steps and learning rate. You can select the JSON files with annotated suffix names to train the different models.

```python
# Download the pretrained model and edit [sr|sample]_[ddpm|sr3]_[resolution option].json about "resume_state":
"resume_state": [your pretrained model's path]
```

### Data Prepare

#### New Start

If you didn't have the data, you can download online ([data] (https://data.caltech.edu/records/czs3p-5ss80)). The training, validation, and test raw data arrays should be sorted as subdirectories of a larger folder. Then you need to change the datasets config to your data path and image resolution: 

```json
"path": { //set the path
        "log": "logs",
        "tb_logger": "tb_logger",
        "dataset": "/shared/ritwik/data/wisodiffusion",
        "results": "/shared/rkurinch/",
        "checkpoint": "checkpoint",
        "resume_state": null
    },
"datasets": {
    "train": {
        "l_resolution": 32, // low resolution need to super_resolution
        "r_resolution": 128, // high resolution
    },
    "val": {
        "l_resolution": 32, // low resolution need to super_resolution
        "r_resolution": 128, // high resolution
    }
},
```

### Training/Resume Training

```python
# Use sr.py and sample.py to train the super resolution task and unconditional generation task, respectively.
# Edit json files to adjust network structure and hyperparameters
python sr.py -p train -c config/sr_sr3.json
```

### Test/Evaluation

```python
# Edit json to add pretrain model path and run the evaluation 
python sr.py -p val -c config/sr_sr3.json

# Quantitative evaluation alone using SSIM/PSNR metrics on given result root
python eval.py -p [result root]
```

### Inference Alone

Set the  image path like steps in `Own Data`, then run the script:

```python
# run the script
python infer.py -c [config file]
```

## Weights and Biases 🎉

The library now supports experiment tracking, model checkpointing and model prediction visualization with [Weights and Biases](https://wandb.ai/site). You will need to [install W&B](https://pypi.org/project/wandb/) and login by using your [access token](https://wandb.ai/authorize). 

```
pip install wandb

# get your access token from wandb.ai/authorize
wandb login
```

W&B logging functionality is added to the `sr.py`, `sample.py` and `infer.py` files. You can pass `-enable_wandb` to start logging.

- `-log_wandb_ckpt`: Pass this argument along with `-enable_wandb` to save model checkpoints as [W&B Artifacts](https://docs.wandb.ai/guides/artifacts). Both `sr.py` and `sample.py` is enabled with model checkpointing. 
- `-log_eval`: Pass this argument along with `-enable_wandb` to save the evaluation result as interactive [W&B Tables](https://docs.wandb.ai/guides/data-vis). Note that only `sr.py` is enabled with this feature. If you run `sample.py` in eval mode, the generated images will automatically be logged as image media panel. 
- `-log_infer`: While running `infer.py` pass this argument along with `-enable_wandb` to log the inference results as interactive W&B Tables.
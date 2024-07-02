# VideoTraining

## Installation


Clone the repo:
```
git clone --recurse-submodules https://github.com/IDGallagher/VideoTraining.git
```


Install ffmpeg:
```
apt-get update; apt-get install ffmpeg
```

```
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install
rm -Rf aws
```


Copy secrets.dat to your home directory with the following variables:

```
export WANDB_API_KEY=""
```

To set up WANDB:
```
pip install wandb; source ~/secrets.dat; wandb login $WANDB_API_KEY;
```


## Downloading datasets

Install our version of video2dataset:
```
cd VideoTraining/video2dataset; python -m venv .venv; . .venv/bin/activate; pip install -r requirements.txt; pip install -e .; mkdir ../tmp;
```

## Training

Install miniconda:
```
# Setup Ubuntu
apt-get update --yes
apt-get upgrade --yes
apt-get install awscli

# Get Miniconda and make it the main Python interpreter
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda 
rm ~/miniconda.sh

export PATH=~/miniconda/bin:$PATH
```

Create environment from AnimateDiff/environment.yaml
```
cd AnimateDiff; conda env create -f environment.yaml
```

```
conda activate animatediff
```

```
cd ~; mkdir .config
```
Copy aws.zip to ~ then and copy fsspec.zip to ~/.config and unzip both
```
cd models; rm -Rf StableDiffusion; cd ..
git lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 models/StableDiffusion/
```

```
python ./AnimateDiff/train.py --config "./configs/ad-training.yaml"
```

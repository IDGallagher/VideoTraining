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


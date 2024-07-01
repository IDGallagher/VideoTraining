# VideoTraining

## Installation
```
git clone --recurse-submodules https://github.com/IDGallagher/VideoTraining.git
```

Copy secrets.dat to your home directory with the following variables:

```
export WANDB_API_KEY="";export GIT_USERNAME="";export GIT_EMAIL=""
```

Run the code below and configure github with the generated ssh key to connect to the repo:
```
!source ~/secrets.dat; git remote set-url origin git@github.com:IDGallagher/VideoTraining.git; git config --global user.name $GIT_USERNAME; git config --global user.email $GIT_EMAIL; ssh-keygen -t rsa -C $GIT_EMAIL -N '' -f ~/.ssh/id_rsa <<< y; tail ~/.ssh/id_rsa.pub;
```

Run the following commands to commit changes and push:
```
git add --all && git commit -m "runpod changes"
git push
```

To set up WANDB:
```
!pip install wandb; source ~/secrets.dat; wandb login $WANDB_API_KEY;
```

## Downloading datasets

Install our version of video2dataset:
```
!cd video2dataset; python -m venv .venv; . .venv/bin/activate; pip install -r requirements.txt; pip install -e .; mkdir ../tmp;
```


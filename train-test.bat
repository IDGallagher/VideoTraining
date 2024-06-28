call %windir%\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy ByPass -NoExit -Command "& 'C:\Users\user\miniconda3\shell\condabin\conda-hook.ps1' ; conda activate animatediff " 

call python ./AnimateDiff/train.py --wandb --config ".\configs\ad-training.yaml"

call conda deactivate
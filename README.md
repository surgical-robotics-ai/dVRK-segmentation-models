# Surgical segmentation models

## Dependencies

```bash
sudo apt install ffmpeg
```
## Installation of anaconda environment

```bash
conda create -n surg_env python=3.9 numpy ipython  -y && conda activate surg_env
```
## Installation of `surg_seg` package

`Surg_seg` is a python package that includes most of the code to interface with the trained models

```bash
pip install -e . -r requirements.txt --user
```

## Erase Anaconda virtual environment

```bash
conda remove --name surg_env --all
conda info --envs
```
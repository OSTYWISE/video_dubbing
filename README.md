# Video dubbing with Audio aligner project

<p align="center">
  <a href="#about">About</a> •
  <a href="#audioaligner">AudioAligner</a> •
  <a href="#work-process">Work process</a> •
  <a href="#videodubber">VideoDubber</a> •
  <a href="installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#possible-problems">Possible problems</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains the code for video dubbing. But the main contribution of this repo is the AudioAligner architecture.

## AudioAligner

This architecture takes translated speech from Text-To-Speech model (Vocoder) and align it to original audio with speech.

The model has 2 targets (weighted multiple loss) in order to:
1. Align translated audio to original one (alignment, style, intonation, etc.)
2. Keep the information from the audio as in translated text (great recognizable speech as output)

For the second problem I use the idea of distillation and train the model to return audios that could be transcribed (with STT model) to correct translated text.

Unlike conventional DL models, there are 2 options to get result:
- Apply untrained model in a single-batch regime (overfitting is not bad)
- Training a large model

## Work process

The project is currently in its early stages of development. Below are the ground stages of the creation process:
1. Write fully functional baseline architecture of AudioAligner and apply it to one-batch-example -> first conclusions
2. Build modern architecture for AudioAligner -> provide results
3. Build VideoDuubber that works with narrative video
4. Build VideoDubber that works with large diverse vidoes like movies


## VideoDubber

VideoDubber consists of many sequential pre-trained models:
1. Audio/Speech Specification and Separation (distinct people, background sounds, music)
2. Speech-To-Text (STT) model
3. Tranlsator
4. Text-To-Speech (TTS) model
5. Voice cloning
6. Aligner


## Installation

Installation may depend on your task. The general steps are the following:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```
3. In order to avoid any problems working with audio and video [ffmpeg](https://www.ffmpeg.org/) is necessary to install.
To handle problems: [Install ffmpeg guide for MacOS](https://youtu.be/8UV7QG0DZLM?si=4vZb2oJDJBIsEUCA).

4. The project is exploiting MLOps platforms: [CometML](https://www.comet.com/) or [Wandb](https://wandb.ai/site/).
You should create your account, get api_key and create variable COMETML_API_KEY or WANDB_API_KEY to be able to run train.py.


## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

### Possible problems

```python
whisper.load_model(’tiny’) -> AttributeError: module 'whisper' has no attribute 'load_model'
- Install using command: pip install git+https://github.com/openai/whisper.git
```

Translation model documentation: https://github.com/facebookresearch/fairseq/tree/nllb?tab=readme-ov-file

## Credits

This template for repo is taken from [pytorch_project_template](https://github.com/Blinorot/pytorch_project_template) repository

Part of the code is adapted from https://github.com/am-sokolov/videodubber?tab=readme-ov-file


## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

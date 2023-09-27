# Woodpecker: Hallucination Correction for MLLMs 🔧

<p align="center">
    <img src="./assets/name.png" width="90%" height="90%">
</p>

<font size=7><div align='center' > <a href="https://3f42ced93bd85313af.gradio.live/">**Online Demo**</a> | <a href=https://github.com/BradyFU/Hallucination-Correction-for-MLLMs>**Paper [Coming Soon]**</a> </div></font>

<p align="center">
    <img src="./assets/framework.jpg" width="96%" height="96%">
</p>

This is the first work to correct hallucination in multimodal large language models. If you have any question, please feel free to email bradyfu24@gmail.com or add weChat ID xjtupanda.

## News
- [09-26] We release our code and the online demo. The paper will be coming soon! 🔥🔥🔥

## Demo
Please feel free to try our [Online Demo](https://3f42ced93bd85313af.gradio.live/)!

<p align="center">
<img src="./assets/example.jpeg" width="96%" height="96%">
</p>

## Preliminary

1. Create conda environment

```bash
conda create -n corrector python=3.10
conda activate 
pip -r requirements.txt
```

2. Install required packages and models

- Install `spacy` and relevant model packages, following the instructions in [Link](https://github.com/explosion/spaCy). This is used for some text processing operations.

```bash
pip install -U spacy
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_md
python -m spacy download en_core_web_sm
```
- For our **Open-set Detector**. Install GroundingDINO following the instructions in [Link](https://github.com/IDEA-Research/GroundingDINO).

## Usage

**1. Inference**

To make corrections based on an image and a text output from MLLM, run the inference code as follows:

```Shell
python inference.py \
        --image-path {path/to/image} \
        --text "Some text to be corrected." \
        --detector-config "path/to/GroundingDINO_SwinT_OGC.py" \
        --detector-model "path/to/groundingdino_swint_ogc.pth" \
        --api-key "sk-xxxxxxx" \

```
The output text will be printed in the terminal, and intermediate results saved by default as ```./intermediate_view.json```.

***

**2. Demo setup**

We use mPLUG-Owl as our default MLLM in experiments. If you wish to replicate the online demo, please clone the [project](https://github.com/X-PLUG/mPLUG-Owl) and modify the variables in https://github.com/BradyFU/Woodpecker/blob/e3fcac307cc5ff5a3dc079d9a94b924ebcdc2531/gradio_demo.py#L7 and  https://github.com/BradyFU/Woodpecker/blob/e3fcac307cc5ff5a3dc079d9a94b924ebcdc2531/gradio_demo.py#L35-L36

Then simply run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python gradio_demo.py
```
Here we put the corrector components on GPU with id 0 and mPLUG-Owl on GPU with id 1.



## Acknowledgement
This repository benefits from [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [BLIP-2](https://huggingface.co/Salesforce/blip2-flan-t5-xxl), and [LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter). Thanks for their awesome works.


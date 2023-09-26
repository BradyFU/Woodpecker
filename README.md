# Hallucination-Correction-for-MLLMs

[[Demo]](https://05f3c84de881927054.gradio.live)

**Note**: The online demo is temporary and subject to regular updates for now.

<p align="center">
    <a href="https://05f3c84de881927054.gradio.live"><img src="assets/framework.png" width="90%"></a> <br> Our framework in detail with a real case.
</p>

## News
- [09/26] 🚀 We released our code and the online demo.

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

We use mPLUG-Owl as our default MLLM in experiments. If you wish to replicate the online demo, please clone the [project](https://github.com/X-PLUG/mPLUG-Owl) and modify the variables in https://github.com/BradyFU/Hallucination-Correction-for-MLLMs/blob/c6826d82785828673064d73d9722fd71214f4b3c/gradio_demo.py#L7 and  https://github.com/BradyFU/Hallucination-Correction-for-MLLMs/blob/c6826d82785828673064d73d9722fd71214f4b3c/gradio_demo.py#L35-L36

Then simply run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python gradio_demo.py
```
Here we put the corrector components on GPU with id 0 and mPLUG-Owl on GPU with id 1.



## Acknowledgement
- [BLIP-2](https://huggingface.co/Salesforce/blip2-flan-t5-xxl) We adopt a blip2-flan-t5-xxl as our VQA model.
- [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl) The baseline MLLM we use in experiments. Thanks for the code and the great work.
- [ZeroFEC](https://github.com/khuangaf/ZeroFEC) An inspiring work. We adopt their QA-to-Claim model and refer to part of their code.
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) The open-set object detector we use. Thanks for the great work.
- [LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter) We refer to part of the code for the demo building.

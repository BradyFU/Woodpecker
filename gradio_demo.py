import os
import torch
from transformers import AutoTokenizer
from vis_corrector import Corrector
from models.utils import extract_boxes, find_matching_boxes, annotate
import sys
sys.path.append('path/to/mPLUG-Owl')
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

from PIL import Image
from types import SimpleNamespace
import cv2
import gradio as gr
import uuid



# ========================================
#             Model Initialization
# ========================================
PROMPT_TEMPLATE = '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <image>
Human: {question}
AI: '''

print('Initializing demo')

# initialize corrector
args_dict = {
        'api_key': "sk-xxxxxxxxxxxxxxxx",
        'api_base': "https://api.openai.com/v1",
        'val_model_path': "Salesforce/blip2-flan-t5-xxl",
        'qa2c_model_path': "khhuang/zerofec-qa2claim-t5-base",
        'detector_config':"path/to/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        'detector_model_path':"path/to/GroundingDINO/weights/groundingdino_swint_ogc.pth",
        'cache_dir': './cache_dir',
}

model_args = SimpleNamespace(**args_dict)
corrector = Corrector(model_args)


# initialize mplug-Owl
pretrained_ckpt = "MAGAer13/mplug-owl-llama-7b"
model = MplugOwlForConditionalGeneration.from_pretrained(
    pretrained_ckpt,
    torch_dtype=torch.bfloat16,
).to("cuda:1")
image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
processor = MplugOwlProcessor(image_processor, tokenizer)

print('Initialization Finished')

@torch.no_grad()
def my_model_function(image, question):
    # create a temp dir to save uploaded imgs
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    unique_filename = str(uuid.uuid4()) + ".png"
    
    temp_file_path = os.path.join(temp_dir, unique_filename)
    
    success = cv2.imwrite(temp_file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    try:
        output_text, output_image = model_predict(temp_file_path, question) 
    finally:
        # remove temporary files.
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    
    return output_text, output_image

def get_owl_output(img_path, question):
    prompts = [PROMPT_TEMPLATE.format(question=question)]
    image_list = [img_path]

    # get response
    generate_kwargs = {
        'do_sample': False,
        'top_k': 5,
        'max_length': 512
    }
    images = [Image.open(_) for _ in image_list]
    inputs = processor(text=prompts, images=images, return_tensors='pt')
    inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        res = model.generate(**inputs, **generate_kwargs)
    sentence = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
    return sentence

def model_predict(image_path, question):
    
    temp_output_filepath = os.path.join("temp", str(uuid.uuid4()) + ".png")
    os.makedirs("temp", exist_ok=True)
    try:
        owl_output = get_owl_output(image_path, question)
        corrector_sample = {
            'img_path': image_path,
            'input_desc': owl_output
        }
        corrector_sample = corrector.correct(corrector_sample)
        corrector_output = corrector_sample['output']
        
        extracted_boxes = extract_boxes(corrector_output)
        boxes, phrases = find_matching_boxes(extracted_boxes, corrector_sample['entity_info'])
        output_image = annotate(image_path, boxes, phrases)
        cv2.imwrite(temp_output_filepath, output_image)
        output_image_pil = Image.open(temp_output_filepath)
        output_text = f"mPLUG-Owl:\n{owl_output}\n\nCorrector:\n{corrector_output}"

        return output_text, output_image_pil
    finally:
        if os.path.exists(temp_output_filepath):
            os.remove(temp_output_filepath)

def create_multi_modal_demo():
    with gr.Blocks() as instruct_demo:
        with gr.Row():
            with gr.Column():
                img = gr.Image(label='Upload Image')
                question = gr.Textbox(lines=2, label="Prompt")

                run_botton = gr.Button("Run")

            with gr.Column():
                output_text = gr.Textbox(lines=10, label="Output")
                output_img = gr.Image(label="Output Image", type='pil')
                
        inputs = [img, question]
        outputs = [output_text, output_img]
        
        examples = [
            ["./examples/case1.jpg", "How many people in the image?"],
            ["./examples/case2.jpg", "Is there any car in the image?"],
            ["./examples/case3.jpg", "Describe this image."],
        ]

        gr.Examples(
            examples=examples,
            inputs=inputs,
            outputs=outputs,
            fn=my_model_function,
            cache_examples=False,
            run_on_click=True
        )
        run_botton.click(fn=my_model_function,
                         inputs=inputs, outputs=outputs)
    return instruct_demo

description = """
# Woodpecker: Hallucination Correction for MLLMs🔧
**Note**: Due to network restrictions, it is recommended that the size of the uploaded image be less than 1M.
"""

with gr.Blocks(css="h1,p {text-align: center;}") as demo:
    gr.Markdown(description)
    with gr.TabItem("Multi-Modal Interaction"):
        create_multi_modal_demo()


demo.queue(api_open=False).launch(share=True)



from vis_corrector import Corrector
from types import SimpleNamespace
import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hallucination Correction for MLLMs.')
    parser.add_argument('--image-path', type=str, help="file path for the text to be corrected.")
    parser.add_argument('--text', type=str, help="text from MLLM to be corrected")
    parser.add_argument('--cache-dir', type=str, help="dir for caching intermediate image",
                        default='./cache_dir')
    
    parser.add_argument('--detector-config', type=str, help="Path to the detector config, \
                        in the form of 'path/to/GroundingDINO_SwinT_OGC.py' ")
    parser.add_argument('--detector-model', type=str, help="Path to the detector checkpoint, \
                        in the form of 'path/to/groundingdino_swint_ogc.pth' ")
    
    parser.add_argument('--api-key', type=str, help="API key for GPT service.")
    parser.add_argument('--api-base', type=str, help="API base link for GPT service.")
    args = parser.parse_args()
    
    args_dict = {
        'api_key': args.api_key if args.api_key else "",
        'api_base': args.api_base if args.api_base else "https://api.openai.com/v1",
        'val_model_path': "Salesforce/blip2-flan-t5-xxl",
        'qa2c_model_path': "khhuang/zerofec-qa2claim-t5-base",
        'detector_config':args.detector_config,
        'detector_model_path':args.detector_model,
        'cache_dir': args.cache_dir,
}

    model_args = SimpleNamespace(**args_dict)

    corrector = Corrector(model_args)

    sample = {
    'img_path': args.image_path,
    'input_desc': args.text
    }
    
    corrected_sample = corrector.correct(sample)
    print(corrected_sample['output'])
    with open('intermediate_view.json', 'w', encoding='utf-8') as file:
        json.dump(corrected_sample, file, ensure_ascii=False, indent=4)
    
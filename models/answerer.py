import os
#os.environ["CUDA_VISIBLE_DEVICES"]="6"
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from typing import Dict


def get_answer(processor, model, img, qs):
    inputs = processor(img, qs, return_tensors="pt").to("cuda:0", torch.float16)

    generated_ids = model.generate(**inputs)
    generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text.strip()

def get_all_answers(processor, model, entity_list, qs, ent_info, input_img_path, cur_answers):
    # This should return a dict. Since a question may correspond to multiple instances of a same kind of object.
    # case 1: involve multiple entities or 'where' type question: use the whole img.
    if len(entity_list)>1 or 'where' in qs.lower() or any([ent not in ent_info for ent in entity_list]):
        img = Image.open(input_img_path).convert('RGB')
        answer = get_answer(processor, model, img, qs) 
        cur_answers.setdefault('overall', [])   # use a special category 'overall' to denote answers that involve multiple objects.
        cur_answers['overall'].append((qs, answer))
    else:
        entity = entity_list[0]
        # case 2: single entity : single/multiple instances.
        for idx, img_path in enumerate(ent_info[entity]['crop_path']):
            img = Image.open(img_path).convert('RGB')
            answer = get_answer(processor, model, img, qs)
            cur_answers.setdefault(entity, [])
            if idx + 1 > len(cur_answers[entity]):
                cur_answers[entity].append([])
            cur_answers[entity][idx].append((qs, answer))
    return cur_answers

class Answerer:
    '''
        Input: 
            'generated_questions': a list of 2-ele list, each [qs(str), involved entities(str)]
            'entity_info': A dict recording the global object information.
            key: obj name. (obj1 | obj2 | obj3)
            value:
                {
                    total_count: detected counts of that obj.
                    
                    crop_path: a list of str, denoting the path to cached intermediate file, i.e., cropped out region of that obj.
                        Note: if total_count > 1, may use the whole image in the following steps.
                        
                    bbox: each [x1, y1, x2, y2], normalized coordinates of left-top and right-bottom corners of bounding boxes.
                }
        Output:
            'generated_answers': An 1-d list of dict. Each dict in the list contains all the (qs, ans) tuple for each object instance.
                                {
                                    overall: [(qs, answer), ...]
                                    entity:  [
                                                [(qs, answer), ...]   (for instance 1 of this type of entity)
                                                    ...
                                             ]
                                }
    '''
    
    def __init__(self, args):
        
        val_model_path = args.val_model_path
        self.args = args
        self.processor = Blip2Processor.from_pretrained(val_model_path)
        device_map = {
            "query_tokens": 0,  # a number. used to set to 0.
            "vision_model":0,
            "language_model": 0,
            "language_projection": 0,
            "qformer": 0,
        }
        self.model = Blip2ForConditionalGeneration.from_pretrained(val_model_path, load_in_8bit=True, device_map=device_map, torch_dtype=torch.float16)
    

    def generate_answers(self, sample: Dict):
        generated_qs = sample['generated_questions']
        global_entity_dict = sample['entity_info']
        
        all_answers = []
        for gen_qs in generated_qs:
            # border case: no question asked.
            if len(gen_qs) == 0:
                all_answers.append({})
                continue
            cur_answers = {}
            for cur_qs in gen_qs:
                qs, entity = cur_qs # qs is a str. entity is also a str. may contain multiple entity connected by periods.
                entity_list = entity.split('.')
                entity_list = [e.strip() for e in entity_list if e.strip()]
                
                cur_answers = get_all_answers(self.processor, self.model, entity_list, qs, global_entity_dict, sample['img_path'], cur_answers)
            all_answers.append(cur_answers)
                
        sample['generated_answers'] = all_answers
        return sample
    
import os

from typing import Dict
from tqdm import tqdm
from PIL import Image
import numpy as np
from collections import defaultdict
import shortuuid
from torchvision.ops import box_convert
import torch
from models.utils import compute_iou
from groundingdino.util.inference import load_model, load_image, predict
from PIL import Image
import spacy





BOX_TRESHOLD = 0.35     # used in detector api.
TEXT_TRESHOLD = 0.25    # used in detector api.
AREA_THRESHOLD = 0.001   # used to filter out too small object.
IOU_THRESHOLD = 0.95     # used to filter the same instance. greater than threshold means the same instance

def in_dict(ent_dict, norm_box):
    if(len(ent_dict)==0):
        return False
    for ent, ent_info in ent_dict.items():
        if 'bbox' in ent_info and any([compute_iou(norm_box, box) > IOU_THRESHOLD for box in ent_info['bbox']]):
            return True
    return False
    
def extract_detection(global_entity_dict, boxes, phrases, image_source, cache_dir, sample):
        
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    normed_xyxy = np.around(np.clip(xyxy / np.array([w, h, w, h]), 0., 1.), 3).tolist()
    
    os.makedirs(cache_dir, exist_ok=True)
    
    for entity, box, norm_box in zip(phrases, xyxy, normed_xyxy):
        # filter out too small object
        thre = sample['area_threshold'] if 'area_threshold' in sample else AREA_THRESHOLD
        if (norm_box[2]-norm_box[0]) * (norm_box[3]-norm_box[1]) < thre:
            continue
        # filter out object already in the dict
        if in_dict(global_entity_dict, norm_box):
           continue 
       
        # add instance, including the cropped_pic & its original bbox
        crop_id = shortuuid.uuid()
        crop_img = Image.fromarray(image_source).crop(box)
        crop_path = os.path.join(cache_dir, f"{crop_id}.png")
        crop_img.save(crop_path)
        
        global_entity_dict[entity]['total_count'] += 1
        global_entity_dict[entity]['crop_path'].append(crop_path)
        global_entity_dict[entity]['bbox'].append(norm_box)    # [x1, y1, x2, y2] coordinate of left-top and right-bottom corner
        
    return global_entity_dict

def find_most_similar_strings(nlp, source_strings, target_strings):
    
    target_docs = [nlp(text) for text in target_strings]

    def find_most_similar(source_str):
        source_doc = nlp(source_str)
        similarities = [(target_doc, target_doc.similarity(source_doc)) for target_doc in target_docs]
        most_similar_doc = max(similarities, key=lambda item: item[1])[0]
        return most_similar_doc.text

    result = [find_most_similar(source_str) for source_str in source_strings]
    
    return result
        
class Detector:
    '''
        Input: 
            img_path: str.
            named_entity: A list of str. Each in a format: obj1.obj2.obj3...
        Output:
            A list of dict, each dict corresponds to a series of objs.
            key: obj name. (obj1 | obj2 | obj3)
            value:
                {
                    total_count: detected counts of that obj.
                    
                    crop_path: a list of str, denoting the path to cached intermediate file, i.e., cropped out region of that obj.
                        Note: if total_count > 1, may use the whole image in the following steps.
                }
    '''
    def __init__(self, args):
        
        self.model = load_model(args.detector_config, args.detector_model_path, device='cuda:0')
        self.cache_dir = args.cache_dir
        self.args = args
        self.nlp = spacy.load("en_core_web_md")
        
    def detect_objects(self, sample: Dict):
        img_path = sample['img_path']
        extracted_entities = sample['named_entity']
        image_source, image = load_image(img_path)
        
        global_entity_dict = {} # key=entity type name. value = {'total_count':int, 'crop_path':list, 'bbox':list of list(4-ele).}
        global_entity_list = [] # save all the entity type name for each sentence.
        for entity_str in extracted_entities:
            # border case: nothing to extract
            if 'none' in entity_str.lower():
                continue
            entity_list = entity_str.split('.')
            for ent in entity_list:
                global_entity_dict.setdefault(ent, {}).setdefault('total_count', 0)
                global_entity_dict.setdefault(ent, {}).setdefault('crop_path', [])
                global_entity_dict.setdefault(ent, {}).setdefault('bbox', [])
                
            global_entity_list.append(entity_list)
        
            boxes, logits, phrases = predict(
                model=self.model,
                image=image,
                caption=entity_str,
                box_threshold=sample['box_threshold'] if 'box_threshold' in sample else BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD,
                device='cuda:0'
            )
            phrases = find_most_similar_strings(self.nlp, phrases, entity_list)    
            global_entity_dict = extract_detection(global_entity_dict, boxes, phrases, image_source, self.cache_dir, sample)
            
        sample['entity_info'] = global_entity_dict
        sample['entity_list'] = global_entity_list
        return sample

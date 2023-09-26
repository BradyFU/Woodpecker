from models.preprocessor import PreProcessor
from models.entity_extractor import EntityExtractor
from models.detector import Detector
from models.questioner import Questioner
from models.answerer import Answerer
from models.claim_generator import ClaimGenerator
from models.refiner import Refiner
from tqdm import tqdm
from typing import List, Dict
import time

class Corrector:
    def __init__(self, args) -> None:
        # init all the model
        
        self.preprocessor = PreProcessor(args)
        self.entity_extractor = EntityExtractor(args)
        self.detector = Detector(args)
        self.questioner = Questioner(args)
        self.answerer = Answerer(args)
        self.claim_generator = ClaimGenerator(args)
        self.refiner = Refiner(args)
        
        print("Finish loading models.")

    
    def correct(self, sample: Dict):
        '''
        sample is Dict containing at least two fields:
            'input_desc': A passage that contains a description of the image.
            'input_img': Path to a local image 
        '''

        sample = self.preprocessor.generate_sentences(sample)
        sample = self.entity_extractor.extract_entity(sample)
        sample = self.detector.detect_objects(sample)
        sample = self.questioner.generate_questions(sample)
        sample = self.answerer.generate_answers(sample)
        sample = self.claim_generator.generate_claim(sample)
        sample = self.refiner.generate_output(sample)
        
        return sample

    def batch_correct(self, samples: List[Dict]):

        return [self.correct(sample) for sample in tqdm(samples, total=len(samples))]

if __name__ == '__main__':
    from types import SimpleNamespace
    import json
    model_args = {
        'api_key': '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',
        'api_base': 'https://api.openai-proxy.com/v1',
        'val_model_path': '/data/xjtupanda/experiments/MM/projects/Hallu/BLIP/t5_ckpt',
        'qa2c_model_path':'/data/xjtupanda/experiments/MM/projects/Hallu/ZeroFEC-master/ckpt/zerofec-qa2claim-t5-base',
        'detector_config':"/data/xjtupanda/experiments/MM/projects/Hallu/GroundingDINO-main/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        'detector_model_path':"/data/xjtupanda/experiments/MM/projects/Hallu/GroundingDINO-main/weights/groundingdino_swint_ogc.pth",
        'cache_dir': './cache_dir',
}

    model_args = SimpleNamespace(**model_args)

    corrector = Corrector(model_args)
    start_time = time.time()

    sample = {
    'img_path': "test_imgs/000000000034.jpg",
    'input_desc': '''The scene features a lush green field filled with tall grass where a young zebra is grazing on the ground. In the field, there are several other zebras spread out and enjoying the greenery. Some zebras are closer to the camera while others are further away or partially hidden behind grass.
There are a total of six zebras in the image, with one zebra being the focus point. They all appear to be contentedly eating grass in this picturesque setting.'''
    }
    corrected_sample = corrector.correct(sample)
    with open('debug_gpu.json', 'w', encoding='utf-8') as file:
        json.dump(corrected_sample, file, ensure_ascii=False, indent=4)
    
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"代码运行时间: {execution_time} 秒")
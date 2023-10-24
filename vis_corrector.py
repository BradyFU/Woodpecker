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
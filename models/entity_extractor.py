from typing import  Dict
from tqdm import tqdm
import openai
import time

NUM_SECONDS_TO_SLEEP = 0.5
PROMPT_TEMPLATE='''Given a sentence, extract the entities within the sentence for me. 
Extract the common objects and summarize them as general categories without repetition, merge essentially similar objects.
Avoid extracting abstract or non-specific entities. 
Extract entity in the singular form. Output all the extracted types of items in one line and separate each object type with a period. If there is nothing to output, then output a single "None".

Examples:
Sentence:
The image depicts a man laying on the ground next to a motorcycle, which appears to have been involved in a crash.

Output:
man.motorcycle

Sentence:
There are a few people around, including one person standing close to the motorcyclist and another person further away.

Output:
person.motorcyclist

Sentence:
No, there is no car in the image.

Output:
car

Sentence:
The image depicts a group of animals, with a black dog, a white kitten, and a gray cat, sitting on a bed.

Output:
dog.cat.bed

Sentence:
{sentence}

Output:'''

def get_res(sent: str, max_tokens: int=1024):
    content = PROMPT_TEMPLATE.format(sentence=sent)
    while True:
        try:
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=[{
                    'role': 'system',
                    'content': 'You are a language assistant that helps to extract information from given sentences.'
                }, {
                    'role': 'user',
                    'content': content,
                }],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            break
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    res = response['choices'][0]['message']['content']
    return res

class EntityExtractor:
    def __init__(self, args):
        openai.api_key = args.api_key
        openai.api_base = args.api_base
        
        self.args = args
        
    
    def extract_entity(self, sample: Dict):
        extracted_entities = []
        for sent in sample['split_sents']:
            entity_str = get_res(sent)
            extracted_entities.append(entity_str)
        sample['named_entity'] = extracted_entities
        
        return sample
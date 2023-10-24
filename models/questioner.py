from typing import Dict
from tqdm import tqdm
import openai
import time
import spacy

# Do not ask questions related to position or position relationship.
NUM_SECONDS_TO_SLEEP = 0.5
PROMPT_TEMPLATE='''Given a sentence and some entities connnected by periods, you are required to ask some relevant questions about the specified entities involved in the sentence, so that the questions can help to verify the factuality of the sentence.
Questions may involve basic attributes such as colors, actions mentioned in the sentence. Do not ask questions involving object counts or the existence of object.
When asking questions about attributes, try to ask simple questions that only involve one entity. 
Ask questions that can be easily decided visually. Do not ask questions that require complex reasoning.
Do not ask semantically similar questions. Do not ask questions only about scenes or places.
Do not ask questions about uncertain or conjecture parts of the sentence, for example, the parts described with "maybe" or "likely", etc.
It is no need to cover all the specified entities. If there is no question to ask, simply output a 'None'.
When asking questions, do not assume the claims in the description as true in advance. Only ask questions relevant to the information in the sentence.
Only ask questions about common, specific and concrete entities. The entities involved in the questions are limited to the range within the given entities.
Output only one question in each line. For each line, first output the question, then a single '&', and finally entities involved in the question, still connected by periods if multiple entities are involved. 

Examples:
Sentence:
There are one black dog and two white cats in the image.

Entities:
dog.cat

Questions:
What color is the cat?&cat
What color is the dog?&dog

Sentence:
The man is wearing a baseball cap and appears to be smoking.

Entities:
man

Questions:
What is the man wearing?&man
What is the man doing?&man

Sentence:
The image depicts a busy kitchen, with a man in a white apron. The man is standing in the middle of the kitchen.

Entities:
kitchen.man

Questions:
What does the man wear?&man
Is the man standing in the middle of the kitchen?&man.kitchen

Sentence:
{sent}

Entities:
{entity}

Questions:'''


def remove_duplicates(res):
    qs_set = set()
    output = []
    for s in res:
        qs, ent = s
        if qs in qs_set:
            continue
        else:
            output.append(s)
            qs_set.add(qs)
    return output

def get_res(nlp, entity: str, sent: str, max_tokens: int=1024):
    content = PROMPT_TEMPLATE.format(sent=sent, entity=entity)
    while True:
        try:
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=[{
                    'role': 'system',
                    'content': 'You are a language assistant that helps to ask questions about a sentence.'
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

    res = response['choices'][0]['message']['content'].splitlines()
    res = [s.split('&') for s in res if s.lower() != 'none']
    entity_list = entity.split('.')
    
    res = [s for s in res if len(s)==2]
    res = remove_duplicates(res)
    res = [s for s in res if set(s[1].split('.')).issubset(set(entity_list)) ]
    return res

class Questioner:
    '''
        Input:
            For each splitted sentences:
                A sentence and list of existent objects. (only questions about existent objects)
        Output:
            For each splitted sentences:
                A list of 2-ele list: [[question, involved object type], [qs, obj], ...]         
    '''
    def __init__(self, args):
        
        openai.api_key = args.api_key
        openai.api_base = args.api_base
        self.args = args
    
        self.nlp = spacy.load("en_core_web_sm")
        
    def generate_questions(self, sample: Dict):
        sentences = sample['split_sents']
        global_entity_dict = sample['entity_info']
        global_entity_list = sample['entity_list']
        
        qs_list = []
        for ent_list, sent in zip(global_entity_list, sentences):
            exist_entity = [ent for ent in ent_list if ent in global_entity_dict and global_entity_dict[ent]['total_count'] > 0]
            
            # border case: no detection result for any entity. no question asked.
            if len(exist_entity)==0 :
                qs_list.append([])
                continue
            
            questions = get_res(self.nlp, '.'.join(exist_entity), sent)
            qs_list.append(questions)
        sample['generated_questions'] = qs_list
        return sample
    
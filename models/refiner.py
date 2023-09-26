from typing import  Dict
import openai
import time

NUM_SECONDS_TO_SLEEP = 0.5
PROMPT_TEMPLATE='''Given a passage and some supplementary information, you are required to correct and output the refined passage in a fluent and natural style, following these rules:
1. The supplementary information may include some of the following parts:
    "Counting" information that specifies how many instances of a certain kind of entity exist, and their associated bounding boxes;
    "Specific" information that describes attribute information specific to each entity instance, including bounding boxes, colors, etc. The information is arranged in the form of "entity 1: [bbox] "info of this entity". Note that the entity in "Specific" information corresponds to that in the "Counting" information.
    "Overall" information that may involve information about multiple entity objects. 
2. Try to retain the original sentence with minimal changes.
3. The number of entitie instances should match the number in the 'Counting' information. Also correct the number counts if the number stated in the original sentence does not match the counting information.
4. If the original sentence is already correct, then just keep it. If you need to rewrite the original sentence, when rewriting, try to modify the original sentence as little as possible based on the original sentence, and use the supplementary information as guidance to correct or enrich the original sentence. 
5. In the refined passage, when describing entities mentioned in the "Specific" supplementary information, add their associated bounding boxes in parentheses right after them, in the form of "entity([bbox])". If multiple entities of the same kind are mentioned, then seperate the box with ';', in the form of "entity([bbox1];[bbox2])"

Examples:
Supplementary information:
Counting:
There are 2 suitcase
suitcase 1: [0.001, 0.307, 0.999, 0.999] The suitcases are black.
suitcase 2: [0.149, 0.0, 0.999, 0.455] The suitcases are brown.

Passage:
The two suitcases in the image are black.

Refined passage:
There are a black suitcase([0.001, 0.307, 0.999, 0.999]) and a brown suitcase([0.149, 0.0, 0.999, 0.455]) in the image.

-------------------

Supplementary information:
Counting: There is no person.

Passage:
Yes, there are two people in the image.

Refined passage: 
No, there are no people in the image.

-------------------

Supplementary information:
Counting: 
There are 5 children.
children 1: [0.001, 0.216, 0.258, 0.935]
children 2: [0.573, 0.251, 0.98, 0.813]
children 3: [0.569, 0.313, 0.701, 0.662]
children 4: [0.31, 0.259, 0.582, 0.775]
children 5: [0.224, 0.283, 0.401, 0.663]

There is no shirts.

There are 1 field.
field 1: [0.001, 0.444, 0.998, 0.997]

There are 2 frisbees.
frisbees 1: [0.392, 0.473, 0.569, 0.713]
frisbees 2: [0.72, 0.486, 0.941, 0.76]

There is no backpack.

There is no handbag.

Specific:
children 1: [0.001, 0.216, 0.258, 0.935] The shirts are red and white. The children are holding a pig. The children are sitting on the grass.
children 2: [0.573, 0.251, 0.98, 0.813] The shirts are blue. The children are holding a frisbee. The children are sitting on the grass.
children 3: [0.569, 0.313, 0.701, 0.662] The shirts are blue. The children are holding a teddy bear. The children are sitting on the grass.
children 4: [0.31, 0.259, 0.582, 0.775] The shirts are brown. The children are holding a frisbee. The children are sitting on the grass.
children 5: [0.224, 0.283, 0.401, 0.663] The shirts are red. The children are holding a ball. The children are sitting on the grass.

frisbees 1: [0.392, 0.473, 0.569, 0.713] The frisbees are white.
frisbees 2: [0.72, 0.486, 0.941, 0.76] The frisbees are white.

Overall:
The children are sitting on the grass.
The children are in the grass.


Passage:
The image shows a group of young children, all wearing black shirts, sitting in a grassy field. They appear to be having a good time as they each hold two white frisbees. 
There's a total of seven children, ranging in age from young toddlers to older children, scattered throughout the scene. Some of the children are standing while others are sitting, enjoying their time in the field. 
In the background, there are several other items, such as a couple of backpacks placed near the field, and a handbag placed further back in the scene.

Refined passage: 
The image shows a group of five children([0.001, 0.216, 0.258, 0.935];[0.573, 0.251, 0.98, 0.813];[0.569, 0.313, 0.701, 0.662];[0.31, 0.259, 0.582, 0.775];[0.224, 0.283, 0.401, 0.663]) sitting in a grassy field([0.001, 0.444, 0.998, 0.997]). The children are wearing black shirts. Some of the children are holding a white frisbee([0.392, 0.473, 0.569, 0.713];[0.72, 0.486, 0.941, 0.76]).
There's a total of five children, ranging in age from young toddlers to older children, scattered throughout the scene. All the children are sitting, enjoying their time in the field. 

-------------------

Supplementary information:
{sup_info}
Passage:
{text}

Refined passage: '''

def get_output(text: str, sup_info: str, max_tokens: int=4096):
    content = PROMPT_TEMPLATE.format(sup_info=sup_info, text=text)
    while True:
        try:
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo-16k',
                messages=[{
                    'role': 'system',
                    'content': 'You are a language assistant that helps to refine a passage according to instructions.'
                }, {
                    'role': 'user',
                    'content': content,
                }],
                temperature=0.2,  
                max_tokens=max_tokens,
            )
            break
        # except openai.error.RateLimitError:
        #     pass
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    return response['choices'][0]['message']['content']

class Refiner:
    '''
        Input:
                'split_sents': 1-d list. Sentences splitted from the passage.
                'claim': 2-d list. Achieve by merging 'generated_questions' and 'generated answers' into sentence-level claims.
        Output:
                'output' : Final output, a refined passage.
    '''
    
    def __init__(self, args):
        openai.api_key = args.api_key
        openai.api_base = args.api_base
        
        self.args = args
    

    def generate_output(self, sample: Dict):
        all_claim = sample['claim']
        global_entity_dict = sample['entity_info']
        
        # three parts: counting, specific, overall
        sup_info = ""
        # add counting info.
        sup_info += all_claim['counting']
        
        # add specific info.
        if 'specific' in all_claim and len(all_claim['specific']) > 0:
            sup_info += "Specific:\n"
            specific_claim_list = []
            for entity, instance_claim in all_claim['specific'].items():
                cur_entity_claim_list = []
                for idx, instance_claim_list in enumerate(instance_claim):
                    cur_inst_bbox = global_entity_dict[entity]['bbox'][idx]
                    cur_entity_claim_list.append(f"{entity} {idx + 1}: {cur_inst_bbox} " + ' '.join(instance_claim_list))
                specific_claim_list.append('\n'.join(cur_entity_claim_list))
            sup_info += '\n\n'.join(specific_claim_list)
            sup_info += '\n\n'
            
        # add overall info.
        if 'overall' in all_claim and len(all_claim['overall']) > 0:
            sup_info += "Overall:\n"
            sup_info += '\n'.join(all_claim['overall'])
            sup_info += '\n\n'
            
        sample['output'] = get_output(sample['input_desc'], sup_info)
        return sample

    
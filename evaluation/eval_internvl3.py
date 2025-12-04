from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from tqdm import tqdm
import os
import json
from utils import build_prompt
import re

def eval_model():
    device = "cuda:0"
    model_name = "InternVL3-2B"

    if model_name == "InternVL3-2B": # can be changed to InternVL3-8B/ InternVL3-14B
        model_checkpoint = "OpenGVLab/InternVL3-2B-hf" 
        processor = AutoProcessor.from_pretrained(model_checkpoint)
        model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map=device, torch_dtype=torch.bfloat16)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")
    
    annotations_path = f'../dataset/annotations.json'

    with open(annotations_path, 'r') as file:
        annotations = json.load(file)
    
    acc = {
         'resemblance':{
                'normal':{'correct':0,'all':0},
                'misleading':{'correct':0,'all':0}
         },
         'representation':{
                'normal':{'correct':0,'all':0},
                'misleading':{'correct':0,'all':0}
         },
         'material':{
                'normal':{'correct':0,'all':0},
                'misleading':{'correct':0,'all':0}
         },
         'mirror':{
                'normal':{'correct':0,'all':0},
                'misleading':{'correct':0,'all':0}
         },
         'occlusion':{
                'normal':{'correct':0,'all':0},
                'misleading':{'correct':0,'all':0}
         },
         'illusion':{
                'normal':{'correct':0,'all':0},
                'misleading':{'correct':0,'all':0}
         },
         'overall':{
                'normal':{'correct':0,'all':0},
                'misleading':{'correct':0,'all':0}
         }
    }

    results = []
    
    for idx, data in enumerate(tqdm(annotations, desc="Evaluating...")):
        image_file = data['image']
        image_path = os.path.join(f'../dataset/', image_file)
        cate = data['category']
        type = data['type']
        acc[cate][type]['all'] += 1
        acc['overall'][type]['all'] += 1
        question = data['question']
        options = data['choices']
        formatted_prompt, label2option, gold_label = build_prompt(question, options)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_path},
                    {"type": "text", "text": formatted_prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
        generate_ids = model.generate(**inputs, do_sample=False,max_new_tokens=5,num_beams=1)
        
        decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        output_text = re.findall(r'\b[A-D]\b', decoded_output)
        output_text = output_text[0]
        out = label2option[output_text]

        if out == data['answer']:
            acc[cate][type]['correct'] += 1
            acc['overall'][type]['correct'] += 1

        results.append(
                {"image":image_file,"is_correct":out == data['answer'],"question":formatted_prompt,
                    "pred_answer": output_text, "gt_answer": data['answer']}
            )
            
    print(f"Evaluation on {model_name}.")

    # Print accuracy results
    for cate in acc:
        print(f"Category: {cate}")
        for type in acc[cate]:
            print(f"Type: {type}")
            correct = acc[cate][type]['correct']
            total = acc[cate][type]['all']
            accuracy = correct / total if total > 0 else 0
            print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

        # compute MVI-Sensitivity
        normal_acc = acc[cate]['normal']['correct'] / acc[cate]['normal']['all']
        misleading_acc = acc[cate]['misleading']['correct'] / acc[cate]['misleading']['all']
        relative_drop = (normal_acc - misleading_acc) / normal_acc if normal_acc > 0 else 0
        print(f"MVI-Sensitivity: {relative_drop:.4f}")

    # with open(f"./{model_name}.json", 'w') as f:
    #     json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    eval_model()


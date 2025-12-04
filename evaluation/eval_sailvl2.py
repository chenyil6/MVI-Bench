import torch
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from tqdm import tqdm
import os
from PIL import Image
import json
from utils import build_prompt
import re


def eval_model():
    device = "cuda:0"
    model_name = "sail2b"

    # load model and processor
    device = "cuda:0"
    if model_name == "sail2b":
        model_path =  "BytedanceDouyinContent/SAIL-VL2-2B"
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
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
        image = Image.open(image_path).convert("RGB")
        cate = data['category']
        type = data['type']
        acc[cate][type]['all'] += 1
        acc['overall'][type]['all'] += 1
        question = data['question']
        options = data['choices']
        formatted_prompt, label2option, gold_label = build_prompt(question, options)

        messages = [
            {"role": "user", "content": [{"type": "image", "image": 'image_path'}, 
            {"type": "text", "text": formatted_prompt}]}
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        inputs = processor(images=image, text=text, return_tensors="pt", padding=True, truncation=True).to(model.device).to(torch.bfloat16)

        generated_ids = model.generate(**inputs, max_new_tokens=512)
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response = response.split('<|im_end|>')[0].strip()

        output_text = re.findall(r'\b[A-D]\b', response)
        result = output_text[0]
        out = label2option[result]
        
        if out == data['answer']:
            acc[cate][type]['correct'] += 1
            acc['overall'][type]['correct'] += 1

        results.append(
                {"image":image_file,"is_correct":out == data['answer'],"question":formatted_prompt,
                    "pred_answer": result, "gt_answer": data['answer']}
            )
        

    print(f"Evaluation on {model_name}.")

    # Print accuracy results
    for cate in acc:
        print(f"Category: {cate}")
        for type in acc[cate]:
            correct = acc[cate][type]['correct']
            total = acc[cate][type]['all']
            accuracy = correct / total if total > 0 else 0
            print(f"Type: {type}, Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        # compute MVI-Sensitivity
        normal_acc = acc[cate]['normal']['correct'] / acc[cate]['normal']['all']
        misleading_acc = acc[cate]['misleading']['correct'] / acc[cate]['misleading']['all']
        relative_drop = (normal_acc - misleading_acc) / normal_acc if normal_acc > 0 else 0
        print(f"MVI-Sensitivity: {relative_drop:.4f}")

    # with open(f"./{model_name}.json", 'w') as f:
    #     json.dump(results, f, indent=4)

if __name__ == "__main__":
    eval_model()

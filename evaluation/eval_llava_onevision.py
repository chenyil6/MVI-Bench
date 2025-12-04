import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from tqdm import tqdm
import os
from PIL import Image
import json
from utils import build_prompt
import re


def eval_model():
    device = "cuda:0"

    model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to(device) 

    processor = AutoProcessor.from_pretrained(model_id)
    
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
        raw_image = Image.open(image_path).convert("RGB")
        cate = data['category']
        type = data['type']
        acc[cate][type]['all'] += 1
        acc['overall'][type]['all'] += 1
        question = data['question']
        options = data['choices']
        formatted_prompt, label2option, gold_label = build_prompt(question, options)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": formatted_prompt},
                    {"type": "image"},

                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(device).to(torch.float16) 

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=5, do_sample=False)

        output_text= processor.decode(output[0][2:], skip_special_tokens=True)
        output_text = output_text.split("\n")[-1]

        output_text = re.findall(r'\b[A-D]\b', output_text)
        result = output_text[0]
        out = label2option[result]
        
        if out == data['answer']:
            acc[cate][type]['correct'] += 1
            acc['overall'][type]['correct'] += 1

        results.append(
                {"image":image_file,"is_correct":out == data['answer'],"question":formatted_prompt,
                    "pred_answer": result, "gt_answer": data['answer']}
            )

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

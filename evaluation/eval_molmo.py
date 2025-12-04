import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from tqdm import tqdm
import os
from PIL import Image
import json
from utils import build_prompt
import re


def eval_model():
    device = "cuda:0"

    processor = AutoProcessor.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device
    )

    # load the model
    model = AutoModelForCausalLM.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device
    )
    
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

        #process the image and text
        inputs = processor.process(
            images=[image],
            text=formatted_prompt
        )

        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        with torch.autocast(device_type=device, enabled=True, dtype=torch.bfloat16):
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>",do_sample=False),
                tokenizer=processor.tokenizer
            )

        
        # only get generated tokens; decode them to text
        generated_tokens = output[0,inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        output_text = re.findall(r'\b[A-D]\b', generated_text)
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

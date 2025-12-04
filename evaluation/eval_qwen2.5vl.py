from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from tqdm import tqdm
import os
import re
import json
from utils import build_prompt
from qwen_vl_utils import process_vision_info


def eval_model():
    device = "cuda:0"
    model_name = "qwen3b"

    if model_name == "qwen3b": # can be changed to qwen2.5-7b/qewen2.5-32b
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", 
            torch_dtype=torch.bfloat16,
            #attn_implementation="flash_attention_2",
            #device_map="auto",
            ).to(device)
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")
    
    annotations_path = '../dataset/annotations.json'

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
                        {
                            "type": "image",
                            "image": image_path,
                            "max_pixels": 512*28*28
                        },
                        {"type": "text", "text": formatted_prompt},
                    ],
                }
            ]
        #Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")
        generated_ids = model.generate(**inputs, do_sample=False,max_new_tokens=5, num_beams=1)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output_text = re.findall(r'\b[A-D]\b', output_text[0])
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

    
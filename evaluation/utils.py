import re
import numpy as np

def build_prompt(question, options: list):
    prompt_options = "\nChoices:\n"
    prompt_end = "\nAnswer with the letter from the given choices directly."
    choice_enumeration = "ABCD"
    shuffle_idx = np.random.permutation(len(options)).tolist() 
    gold_idx = shuffle_idx.index(0) 
    gold_label = choice_enumeration[gold_idx] 
    formatted_options = "\n".join([f"{choice_enumeration[i]}. {options[j]}" for i, j in enumerate(shuffle_idx)])
    label2option = {choice_enumeration[i]: options[j] for i, j in enumerate(shuffle_idx)}
    formatted_prompt = f"{question}{prompt_options}{formatted_options}{prompt_end}"
    return formatted_prompt, label2option, gold_label


def extract_choice_letter(output_text):
    if isinstance(output_text, (list, tuple)):
        s = " ".join(map(str, output_text))
    else:
        s = str(output_text)

    s = s.strip()
    if not s:
        return None

    S = s.upper()

    patterns = [
        r'(?<![A-Z0-9_])([A-D])(?![A-Z0-9_])',  
        r'ANSWER[:\s]*([A-D])',                
        r'\b([A-D])\b',                       
        r'([A-D])(?=\d)',                       
        r'([A-D])'                              
    ]

    for p in patterns:
        m = re.search(p, S)
        if m:
            return m.group(1)

    return None
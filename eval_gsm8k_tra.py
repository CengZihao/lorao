import os
import sys
import re
import fire
import torch
import json 
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

# from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def main(
    load_8bit: bool = True,
    base_model: str = "yahma/llama-7b-hf",
    lora_weights: str = "",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )


    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    temperature=0.9
    top_p=0.8
    top_k=40
    num_beams=2
    max_new_tokens=256
    data_num=300
    eps=1e-6

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        repetition_penalty=1.3, #
    )

    with open('/data/zzh/olora/gsm8k/test.jsonl', 'r') as f:
        lines = f.readlines()[:data_num]

    correct_num = 0
    line_idx = 0
    for line in tqdm(lines):
        data = json.loads(line)
        instruction = data['instruction']
        true_result = data['output'].split('####')[-1].strip()
        true_result = true_result.replace(",", "")
    
        test_input=None

        prompt = prompter.generate_prompt(instruction, test_input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        s = generation_output.sequences[0]
        output = tokenizer.decode(s)

        match = re.search(r'### Response:\n(.*?)(\n###|$)', output, re.DOTALL)
        if match:
            response = match.group(1)
            print("\n####")
            print(response)
            print("####")
            numbers = re.findall(r'-?\d+(?:\.\d+)?(?:,\d+)*', response)
            last_numbers = []
            if numbers:
                # 获取最后一个数字
                last_number = numbers[-1].replace(',', '')
                last_numbers.append(last_number)
            print("true: ", str(true_result))
            print("answer: ", str(last_numbers[0]))
            if abs(float(true_result) - float(last_numbers[0])) < eps:
                correct_num += 1
                print("correct")
            else:
                print("wrong")

        print("current accuracy: ", correct_num / (line_idx + 1))
        line_idx += 1

    print("accuracy: ", correct_num / data_num)

if __name__ == "__main__":
    fire.Fire(main)

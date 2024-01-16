import os
import sys
import re
import fire
import torch
import json 
from mypeft import PeftModel
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

    with open('/data/zzh/olora/boolq/transformed_test.jsonl', 'r') as f:
        lines = f.readlines()[:data_num]

    correct_num = 0
    line_idx = 0
    for line in tqdm(lines):
        data = json.loads(line)
        instruction = None
        true_result = "True" if data['output'] == True else "False"
        test_input = data['input']

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
                is_generation_mode=True,
            )

        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        match = re.search(r'### Response:\n(.*?)(\n###|$)', output, re.DOTALL)
        if match:
            response = match.group(1).replace("</s>", "").strip()
            response = "True" if response == "True" else "False"
            print("\ntrue_res: ", true_result)
            print("response: ", response)
            if true_result == response:
                correct_num += 1
                print("correct")
            else:
                print("wrong")

        print("current accuracy: ", correct_num / (line_idx + 1))
        line_idx += 1

    print("accuracy: ", correct_num / data_num)

if __name__ == "__main__":
    fire.Fire(main)

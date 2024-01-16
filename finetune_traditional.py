import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
import wandb
import random
import numpy as np

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

# import mytransformers
# from mytransformers.src.transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter


def seed_torch(seed=42):
    random.seed(seed)   # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)    # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)   # numpy的随机性
    torch.manual_seed(seed)   # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)   # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False   # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True   # 选择确定性算法
    print("fix seed")

seed_torch()


def train(
    # model/data params
    # base_model: str = "decapoda-research/llama-7b-hf",
    base_model: str = "yahma/llama-7b-hf",  # the only required argument
    general_data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca/olora",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 8,
    # batch_size: int = 32,
    # micro_batch_size: int = 16,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    # learning_rate: float = 4.1,
    cutoff_len: int = 512,
    # val_set_size: int = 16,
    val_set_size: int = 1000,
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "debug",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    # resume_from_checkpoint: str = "output/1211tau_3_gsm",  
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0: #master gpu卡
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"general_data_path: {general_data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1)) #world_size=1
    # print("world_size",world_size) # 1
    ddp = world_size != 1
    if ddp: #ddp=False
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model


    # gsm8k_data = load_dataset("json", data_files="/data/zzh/olora/gsm8k/train.json")
    # print("gsm8k_data", gsm8k_data)

    # wealth_data = load_dataset("json", data_files="/data/zzh/olora/wealth/train.jsonl")
    # print("wealth_data", wealth_data)

    # code_data = load_dataset("json", data_files="/data/zzh/olora/code/train.jsonl")
    # print("code_data", code_data)

    # boolq_data = load_dataset("json", data_files="/data/zzh/olora/boolq/transformed_train.jsonl")
    # print("boolq_data", boolq_data)

    # piqa_data = load_dataset("json", data_files="/data/zzh/olora/piqa/transformed_train.jsonl")
    # print("piqa_data", piqa_data)

    tsa_data = load_dataset("json", data_files="/data/zzh/olora/tsa/train_new.jsonl")
    print("tsa_data", tsa_data)

    # loss: crossentropy (next word in vocabulary)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    # print(tokenizer.eos_token_id)

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            # print("prompt:",prompt)
            # print("len(result[input_ids])",len(result["input_ids"]))
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        # print(tokenized_full_prompt)
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)


    if val_set_size > 0:

        # # gsm8k
        # gsm8k_train_val = gsm8k_data['train'].train_test_split(
        #     test_size=val_set_size, shuffle=True, seed=42
        # )
        # gsm8k_train_data = (
        #     gsm8k_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        # )
        # gsm8k_val_data = (
        #     gsm8k_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        # )

        # # wealth
        # wealth_train_val = wealth_data['train'].train_test_split(
        #     test_size=val_set_size, shuffle=True, seed=42
        # )
        # wealth_train_data = (
        #     wealth_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        # )
        # wealth_val_data = (
        #     wealth_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        # )

        # # code
        # code_train_val = code_data['train'].train_test_split(
        #     test_size=val_set_size, shuffle=True, seed=42
        # )
        # code_train_data = (
        #     code_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        # )
        # code_val_data = (
        #     code_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        # )

        # # boolq
        # boolq_train_val = boolq_data['train'].train_test_split(
        #     test_size=val_set_size, shuffle=True, seed=42
        # )
        # boolq_train_data = (
        #     boolq_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        # )
        # boolq_val_data = (
        #     boolq_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        # )

        # # piqa
        # piqa_train_val = piqa_data['train'].train_test_split(
        #     test_size=val_set_size, shuffle=True, seed=42
        # )
        # piqa_train_data = (
        #     piqa_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        # )
        # piqa_val_data = (
        #     piqa_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        # )

        # tsa
        tsa_train_val = tsa_data['train'].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        tsa_train_data = (
            tsa_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        tsa_val_data = (
            tsa_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )

    else:

        # # gsm8k
        # gsm8k_train_data = gsm8k_data["train"].shuffle().map(generate_and_tokenize_prompt)
        # gsm8k_val_data = None

        # # wealth
        # wealth_train_data = wealth_data["train"].shuffle().map(generate_and_tokenize_prompt)
        # wealth_val_data = None

        # # code
        # code_train_data = code_data["train"].shuffle().map(generate_and_tokenize_prompt)
        # code_val_data = None

        # # boolq
        # boolq_train_data = boolq_data["train"].shuffle().map(generate_and_tokenize_prompt)
        # boolq_val_data = None

        # # piqa
        # piqa_train_data = piqa_data["train"].shuffle().map(generate_and_tokenize_prompt)
        # piqa_val_data = None

        # tsa
        tsa_train_data = tsa_data["train"].shuffle().map(generate_and_tokenize_prompt)
        tsa_val_data = None



    if resume_from_checkpoint:
        print("check if there is available checkpoint")
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if not ddp and torch.cuda.device_count() > 1:
        print("not ddp and torch.cuda.device_count() > 1:")
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    wandb_version = "_" + output_dir.split("lora-alpaca/")[-1]

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)




    # wandb.init(project="debug", name="gsm8k"+wandb_version, reinit=True)
    # gsm8k_output_dir = output_dir + "/gsm8k"
    # gsm8k_trainer = transformers.Trainer(
    #     model=model,
    #     train_dataset=gsm8k_train_data, #
    #     eval_dataset=gsm8k_val_data, #
    #     args=transformers.TrainingArguments(
    #         per_device_train_batch_size=micro_batch_size,
    #         per_device_eval_batch_size=micro_batch_size,
    #         gradient_accumulation_steps=gradient_accumulation_steps,
    #         warmup_steps=0,
    #         num_train_epochs=1,
    #         learning_rate=learning_rate,
    #         fp16=True,
    #         logging_steps=5,
    #         optim="adamw_torch",
    #         evaluation_strategy="steps" if val_set_size > 0 else "no",
    #         save_strategy="steps",
    #         eval_steps=10 if val_set_size > 0 else None,
    #         # eval_steps=200 if val_set_size > 0 else None,
    #         save_steps=40,
    #         output_dir=gsm8k_output_dir,
    #         save_total_limit=1,
    #         load_best_model_at_end=True if val_set_size > 0 else False,
    #         ddp_find_unused_parameters=False if ddp else None,
    #         group_by_length=group_by_length,
    #         report_to="wandb" if use_wandb else None,
    #         run_name="gsm8k"+wandb_version if use_wandb else None,
    #     ),
    #     data_collator=transformers.DataCollatorForSeq2Seq(
    #         tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    #     ),
    # )
    # gsm8k_trainer.train()
    # print("lora.default\n", model.model.model.layers[31].self_attn.v_proj.lora_B["default"].weight)







    # wandb.init(project="debug", name="wealth"+wandb_version, reinit=True)
    # wealth_output_dir = output_dir + "/wealth"
    # wealth_trainer = transformers.Trainer(
    #     model=model,
    #     train_dataset=wealth_train_data, #
    #     eval_dataset=wealth_val_data, #
    #     args=transformers.TrainingArguments(
    #         per_device_train_batch_size=micro_batch_size,
    #         per_device_eval_batch_size=micro_batch_size,
    #         gradient_accumulation_steps=gradient_accumulation_steps,
    #         warmup_steps=0,
    #         num_train_epochs=1,
    #         learning_rate=learning_rate,
    #         fp16=True,
    #         logging_steps=5,
    #         optim="adamw_torch",
    #         evaluation_strategy="steps" if val_set_size > 0 else "no",
    #         save_strategy="steps",
    #         eval_steps=10 if val_set_size > 0 else None,
    #         # eval_steps=200 if val_set_size > 0 else None,
    #         save_steps=40,
    #         output_dir=wealth_output_dir,
    #         save_total_limit=1,
    #         load_best_model_at_end=True if val_set_size > 0 else False,
    #         ddp_find_unused_parameters=False if ddp else None,
    #         group_by_length=group_by_length,
    #         report_to="wandb" if use_wandb else None,
    #         run_name="wealth"+wandb_version if use_wandb else None,
    #     ),
    #     data_collator=transformers.DataCollatorForSeq2Seq(
    #         tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    #     ),
    # )
    # wealth_trainer.train()







    # wandb.init(project="debug", name="code"+wandb_version, reinit=True)
    # code_output_dir = output_dir + "/code"
    # code_trainer = transformers.Trainer(
    #     model=model,
    #     train_dataset=code_train_data, #
    #     eval_dataset=code_val_data, #
    #     args=transformers.TrainingArguments(
    #         per_device_train_batch_size=micro_batch_size,
    #         per_device_eval_batch_size=micro_batch_size,
    #         gradient_accumulation_steps=gradient_accumulation_steps,
    #         warmup_steps=0,
    #         num_train_epochs=1,
    #         learning_rate=learning_rate,
    #         fp16=True,
    #         logging_steps=5,
    #         optim="adamw_torch",
    #         evaluation_strategy="steps" if val_set_size > 0 else "no",
    #         save_strategy="steps",
    #         eval_steps=10 if val_set_size > 0 else None,
    #         # eval_steps=200 if val_set_size > 0 else None,
    #         save_steps=40,
    #         output_dir=code_output_dir,
    #         save_total_limit=1,
    #         load_best_model_at_end=True if val_set_size > 0 else False,
    #         ddp_find_unused_parameters=False if ddp else None,
    #         group_by_length=group_by_length,
    #         report_to="wandb" if use_wandb else None,
    #         run_name="code"+wandb_version if use_wandb else None,
    #     ),
    #     data_collator=transformers.DataCollatorForSeq2Seq(
    #         tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    #     ),
    # )
    # code_trainer.train()








    # wandb.init(project="debug", name="boolq"+wandb_version, reinit=True)
    # boolq_output_dir = output_dir + "/boolq"
    # boolq_trainer = transformers.Trainer(
    #     model=model,
    #     train_dataset=boolq_train_data, #
    #     eval_dataset=boolq_val_data, #
    #     args=transformers.TrainingArguments(
    #         per_device_train_batch_size=micro_batch_size,
    #         per_device_eval_batch_size=micro_batch_size,
    #         gradient_accumulation_steps=gradient_accumulation_steps,
    #         warmup_steps=0,
    #         num_train_epochs=1,
    #         learning_rate=learning_rate,
    #         fp16=True,
    #         logging_steps=5,
    #         optim="adamw_torch",
    #         evaluation_strategy="steps" if val_set_size > 0 else "no",
    #         save_strategy="steps",
    #         eval_steps=10 if val_set_size > 0 else None,
    #         # eval_steps=200 if val_set_size > 0 else None,
    #         save_steps=40,
    #         output_dir=boolq_output_dir,
    #         save_total_limit=1,
    #         load_best_model_at_end=True if val_set_size > 0 else False,
    #         ddp_find_unused_parameters=False if ddp else None,
    #         group_by_length=group_by_length,
    #         report_to="wandb" if use_wandb else None,
    #         run_name="boolq"+wandb_version if use_wandb else None,
    #     ),
    #     data_collator=transformers.DataCollatorForSeq2Seq(
    #         tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    #     ),
    # )
    # boolq_trainer.train()
    # print("lora.default\n", model.model.model.layers[31].self_attn.v_proj.lora_B["default"].weight)
    








    # wandb.init(project="debug", name="piqa"+wandb_version, reinit=True)
    # piqa_output_dir = output_dir + "/piqa"
    # piqa_trainer = transformers.Trainer(
    #     model=model,
    #     train_dataset=piqa_train_data, #
    #     eval_dataset=piqa_val_data, #
    #     args=transformers.TrainingArguments(
    #         per_device_train_batch_size=micro_batch_size,
    #         per_device_eval_batch_size=micro_batch_size,
    #         gradient_accumulation_steps=gradient_accumulation_steps,
    #         warmup_steps=0,
    #         num_train_epochs=1,
    #         learning_rate=learning_rate,
    #         fp16=True,
    #         logging_steps=5,
    #         optim="adamw_torch",
    #         evaluation_strategy="steps" if val_set_size > 0 else "no",
    #         save_strategy="steps",
    #         eval_steps=10 if val_set_size > 0 else None,
    #         # eval_steps=200 if val_set_size > 0 else None,
    #         save_steps=40,
    #         output_dir=piqa_output_dir,
    #         save_total_limit=1,
    #         load_best_model_at_end=True if val_set_size > 0 else False,
    #         ddp_find_unused_parameters=False if ddp else None,
    #         group_by_length=group_by_length,
    #         report_to="wandb" if use_wandb else None,
    #         run_name="piqa"+wandb_version if use_wandb else None,
    #     ),
    #     data_collator=transformers.DataCollatorForSeq2Seq(
    #         tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    #     ),
    # )
    # piqa_trainer.train()
    # print("lora.default\n", model.model.model.layers[31].self_attn.v_proj.lora_B["default"].weight)







    wandb.init(project="debug", name="tsa"+wandb_version, reinit=True)
    tsa_output_dir = output_dir + "/tsa"
    tsa_trainer = transformers.Trainer(
        model=model,
        train_dataset=tsa_train_data, #
        eval_dataset=tsa_val_data, #
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=1,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=5,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=10 if val_set_size > 0 else None,
            # eval_steps=200 if val_set_size > 0 else None,
            save_steps=40,
            output_dir=tsa_output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name="tsa"+wandb_version if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    tsa_trainer.train()
    print("lora.default\n", model.model.model.layers[31].self_attn.v_proj.lora_B["default"].weight)






    print("saving pretrained")
    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)

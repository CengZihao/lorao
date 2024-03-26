import os
import sys
from typing import List
import fire
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
import wandb
import random
import numpy as np
from mypeft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from utils.prompter import Prompter
from my_trainer import MyTrainer



def seed_torch(seed=42):
    random.seed(seed) # Python 的随机性
    os.environ['PYTHONHASHSEED'] = str(seed) # 设置 Python 哈希种子，为了禁止 hash 随机化，使得实验可复现
    np.random.seed(seed) # numpy的随机性
    torch.manual_seed(seed) # torch 的 CPU 随机性，为 CPU 设置随机种子
    torch.cuda.manual_seed(seed) # torch 的 GPU 随机性，为当前 GPU 设置随机种子
    torch.cuda.manual_seed_all(seed) # 使用多 GPU 时需要，为所有 GPU 设置随机种子
    torch.backends.cudnn.benchmark = False # 关闭 benchmark，提高稳定性
    torch.backends.cudnn.deterministic = True # 确保 cudnn 的实现是确定的
    print("fix seed")

seed_torch()



def train(
    base_model: str = "yahma/llama-7b-hf",
    output_dir: str = "./lora-alpaca/olora",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512, # 输入 token 的最大长度
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
    train_on_inputs: bool = False,
    add_eos_token: bool = False,
    group_by_length: bool = False,
    # wandb params
    select_dataset: str = "",
    wandb_project: str = "debug",
    wandb_run_name: str = "",
    wandb_watch: str = "", # options: false | gradients | all
    wandb_log_model: str = "", # options: false | true
    resume_from_checkpoint: str = None, # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca", # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0: #master gpu卡
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
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
            f"select_dataset: {select_dataset}\n"
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
    world_size = int(os.environ.get("WORLD_SIZE", 1)) # WORLD_SIZE 表示进程总数
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} # LOCAL_RANK 表示某一 GPU 卡的编号
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    

    # wandb 配置
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
        # 检查是否提供了 wandb_project 参数 or 环境变量 WANDB_PROJECT 已设置，以决定是否使用 wandb 进行日志记录
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model


    if select_dataset == "gsm8k":
        gsm8k_data = load_dataset("json", data_files="/data/zzh/olora/datasets/gsm8k/train.jsonl")
        print("gsm8k_data", gsm8k_data)
    elif select_dataset == "wealth":
        wealth_data = load_dataset("json", data_files="/data/zzh/olora/datasets/wealth/train.jsonl")
        print("wealth_data", wealth_data)
    elif select_dataset == "code":
        code_data = load_dataset("json", data_files="/data/zzh/olora/datasets/code/train_mod.jsonl")
        print("code_data", code_data)
    elif select_dataset == "boolq":
        boolq_data = load_dataset("json", data_files="/data/zzh/olora/datasets/boolq/transformed_train.jsonl")
        print("boolq_data", boolq_data)
    elif select_dataset == "piqa":
        piqa_data = load_dataset("json", data_files="/data/zzh/olora/datasets/piqa/transformed_train.jsonl")
        print("piqa_data", piqa_data)
    elif select_dataset == "tsa":
        tsa_data = load_dataset("json", data_files="/data/zzh/olora/datasets/tsa/train_new.jsonl")
        print("tsa_data", tsa_data)
    elif select_dataset == "social":
        social_data = load_dataset("json", data_files="/data/zzh/olora/datasets/social/train.jsonl")
        print("social_data", social_data)
    elif select_dataset == "agnews":
        agnews_data = load_dataset("json", data_files="/data/zzh/olora/datasets/agnews/train.jsonl")
        print("agnews_data", agnews_data)
    elif select_dataset == "common":
        common_data = load_dataset("json", data_files="/data/zzh/olora/datasets/common/train.jsonl")
        print("common_data", common_data)
    elif select_dataset == "emotion":
        emotion_data = load_dataset("json", data_files="/data/zzh/olora/datasets/emotion/train.jsonl")
        print("emotion_data", emotion_data)
    elif select_dataset == "omw":
        # omw_data = load_dataset("json", data_files="/data/zzh/olora/datasets/omw/train.jsonl")
        # omw_data = load_dataset("json", data_files="/data/zzh/olora/datasets/piqa/transformed_train.jsonl")
        omw_data = load_dataset("json", data_files="/data/zzh/olora/datasets/wino/train.jsonl")
        print("omw_data", omw_data)
    elif select_dataset == "wino":
        wino_data = load_dataset("json", data_files="/data/zzh/olora/datasets/wino/train.jsonl")
        print("wino_data", wino_data)


    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0 # we want this to be different from the eos token
    )
    tokenizer.padding_side = "left" # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        """
        由 prompt 文本得到分词后的 tokens
        """
        result = tokenizer(
            prompt, # 文本
            truncation=True, # 如果输入 token 的长度超过 max_length，则将其截断到 max_length
            max_length=cutoff_len,
            padding=False, # 不填充
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
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
            ]
                # -100 表示忽略的 token，这里是为了让模型只关注输出部分的 loss
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
        if select_dataset == "gsm8k":
            # gsm8k
            gsm8k_train_val = gsm8k_data['train'].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            gsm8k_train_data = (
                gsm8k_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            gsm8k_val_data = (
                gsm8k_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        elif select_dataset == "wealth":
            # wealth
            wealth_train_val = wealth_data['train'].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            wealth_train_data = (
                wealth_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            wealth_val_data = (
                wealth_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        elif select_dataset == "code":
            # code
            code_train_val = code_data['train'].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            code_train_data = (
                code_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            code_val_data = (
                code_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        elif select_dataset == "boolq":
            # boolq
            boolq_train_val = boolq_data['train'].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            boolq_train_data = (
                boolq_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            boolq_val_data = (
                boolq_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        elif select_dataset == "piqa":
            # piqa
            piqa_train_val = piqa_data['train'].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            piqa_train_data = (
                piqa_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            piqa_val_data = (
                piqa_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        elif select_dataset == "tsa":
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
        elif select_dataset == "social":
            # social
            social_train_val = social_data['train'].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            social_train_data = (
                social_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            social_val_data = (
                social_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        elif select_dataset == "agnews":
            # agnews
            agnews_train_val = agnews_data['train'].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            agnews_train_data = (
                agnews_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            agnews_val_data = (
                agnews_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        elif select_dataset == "common":
            # common
            common_train_val = common_data['train'].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            common_train_data = (
                common_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            common_val_data = (
                common_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        elif select_dataset == "emotion":
            # emotion
            emotion_train_val = emotion_data['train'].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            emotion_train_data = (
                emotion_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            emotion_val_data = (
                emotion_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        elif select_dataset == "omw":
            # omw
            omw_train_val = omw_data['train'].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            omw_train_data = (
                omw_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            omw_val_data = (
                omw_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        elif select_dataset == "wino":
            # wino
            wino_train_val = wino_data['train'].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            wino_train_data = (
                wino_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            wino_val_data = (
                wino_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )

    else:
        if select_dataset == "gsm8k":
            # gsm8k
            gsm8k_train_data = gsm8k_data["train"].shuffle().map(generate_and_tokenize_prompt)
            gsm8k_val_data = None
        elif select_dataset == "wealth":
            # wealth
            wealth_train_data = wealth_data["train"].shuffle().map(generate_and_tokenize_prompt)
            wealth_val_data = None
        elif select_dataset == "code":
            # code
            code_train_data = code_data["train"].shuffle().map(generate_and_tokenize_prompt)
            code_val_data = None
        elif select_dataset == "boolq":
            # boolq
            boolq_train_data = boolq_data["train"].shuffle().map(generate_and_tokenize_prompt)
            boolq_val_data = None
        elif select_dataset == "piqa":
            # piqa
            piqa_train_data = piqa_data["train"].shuffle().map(generate_and_tokenize_prompt)
            piqa_val_data = None
        elif select_dataset == "tsa":
            # tsa
            tsa_train_data = tsa_data["train"].shuffle().map(generate_and_tokenize_prompt)
            tsa_val_data = None
        elif select_dataset == "social":
            # social
            social_train_data = social_data["train"].shuffle().map(generate_and_tokenize_prompt)
            social_val_data = None
        elif select_dataset == "agnews":
            # agnews
            agnews_train_data = agnews_data["train"].shuffle().map(generate_and_tokenize_prompt)
            agnews_val_data = None
        elif select_dataset == "common":
            # common
            common_train_data = common_data["train"].shuffle().map(generate_and_tokenize_prompt)
            common_val_data = None
        elif select_dataset == "emotion":
            # emotion
            emotion_train_data = emotion_data["train"].shuffle().map(generate_and_tokenize_prompt)
            emotion_val_data = None
        elif select_dataset == "omw":
            # omw
            omw_train_data = omw_data["train"].shuffle().map(generate_and_tokenize_prompt)
            omw_val_data = None
        elif select_dataset == "wino":
            # wino
            wino_train_data = wino_data["train"].shuffle().map(generate_and_tokenize_prompt)
            wino_val_data = None


    if resume_from_checkpoint:
        # 从 checkpoint 恢复模型训练
        print("check if there is available checkpoint")
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        ) # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            ) # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()

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




    state_dict = model.state_dict()
    for key, value in state_dict.items():
        print(f"{key}: {value.shape}")
        print(value)




    if select_dataset == "gsm8k":
        wandb.init(project="debug", name="gsm8k"+wandb_version, reinit=True)
        gsm8k_output_dir = output_dir + "/gsm8k"
        gsm8k_trainer = MyTrainer(
            model=model,
            train_dataset=gsm8k_train_data, #
            eval_dataset=gsm8k_val_data, #
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
                save_steps=40,
                output_dir=gsm8k_output_dir,
                save_total_limit=1,
                load_best_model_at_end=True if val_set_size > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=group_by_length,
                report_to="wandb" if use_wandb else None,
                run_name="gsm8k"+wandb_version if use_wandb else None,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
        gsm8k_trainer.train(select_dataset="gsm8k",resume_from_checkpoint=resume_from_checkpoint)
        print("have trained gsm8k")
        print(model.model.model.layers[31].self_attn.v_proj.mylora_B.gsm8k.weight)

    elif select_dataset == "wealth":
        wandb.init(project="debug", name="wealth"+wandb_version, reinit=True)
        wealth_output_dir = output_dir + "/wealth"
        wealth_trainer = MyTrainer(
            model=model,
            train_dataset=wealth_train_data, #
            eval_dataset=wealth_val_data, #
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
                output_dir=wealth_output_dir,
                save_total_limit=1,
                load_best_model_at_end=True if val_set_size > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=group_by_length,
                report_to="wandb" if use_wandb else None,
                run_name="wealth"+wandb_version if use_wandb else None,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
        wealth_trainer.train(select_dataset="wealth",resume_from_checkpoint=resume_from_checkpoint)
        print("have trained wealth")
        print(model.model.model.layers[31].self_attn.v_proj.mylora_B.wealth.weight)

    elif select_dataset == "code":
        wandb.init(project="debug", name="code"+wandb_version, reinit=True)
        code_output_dir = output_dir + "/code"
        code_trainer = MyTrainer(
            model=model,
            train_dataset=code_train_data, #
            eval_dataset=code_val_data, #
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
                output_dir=code_output_dir,
                save_total_limit=1,
                load_best_model_at_end=True if val_set_size > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=group_by_length,
                report_to="wandb" if use_wandb else None,
                run_name="code"+wandb_version if use_wandb else None,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
        code_trainer.train(select_dataset="code",resume_from_checkpoint=resume_from_checkpoint)
        print("have trained code")
        print(model.model.model.layers[31].self_attn.v_proj.mylora_B.code.weight)

    elif select_dataset == "boolq":
        wandb.init(project="debug", name="boolq"+wandb_version, reinit=True)
        boolq_output_dir = output_dir + "/boolq"
        boolq_trainer = MyTrainer(
            model=model,
            train_dataset=boolq_train_data, #
            eval_dataset=boolq_val_data, #
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
                output_dir=boolq_output_dir,
                save_total_limit=1,
                load_best_model_at_end=True if val_set_size > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=group_by_length,
                report_to="wandb" if use_wandb else None,
                run_name="boolq"+wandb_version if use_wandb else None,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
        boolq_trainer.train(select_dataset="boolq",resume_from_checkpoint=resume_from_checkpoint)
        print("have trained boolq")
        print(model.model.model.layers[31].self_attn.v_proj.mylora_B.boolq.weight)
    
    elif select_dataset == "piqa":
        wandb.init(project="debug", name="piqa"+wandb_version, reinit=True)
        piqa_output_dir = output_dir + "/piqa"
        piqa_trainer = MyTrainer(
            model=model,
            train_dataset=piqa_train_data, #
            eval_dataset=piqa_val_data, #
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
                output_dir=piqa_output_dir,
                save_total_limit=1,
                load_best_model_at_end=True if val_set_size > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=group_by_length,
                report_to="wandb" if use_wandb else None,
                run_name="piqa"+wandb_version if use_wandb else None,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
        piqa_trainer.train(select_dataset="piqa",resume_from_checkpoint=resume_from_checkpoint)
        print("have trained piqa")
        print(model.model.model.layers[31].self_attn.v_proj.mylora_B.piqa.weight)

    elif select_dataset == "tsa":
        wandb.init(project="debug", name="tsa"+wandb_version, reinit=True)
        tsa_output_dir = output_dir + "/tsa"
        tsa_trainer = MyTrainer(
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
        tsa_trainer.train(select_dataset="tsa",resume_from_checkpoint=resume_from_checkpoint)
        print("have trained tsa")
        print(model.model.model.layers[31].self_attn.v_proj.mylora_B.tsa.weight)
    
    elif select_dataset == "social":
        wandb.init(project="debug", name="social"+wandb_version, reinit=True)
        social_output_dir = output_dir + "/social"
        social_trainer = MyTrainer(
            model=model,
            train_dataset=social_train_data, #
            eval_dataset=social_val_data, #
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
                output_dir=social_output_dir,
                save_total_limit=1,
                load_best_model_at_end=True if val_set_size > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=group_by_length,
                report_to="wandb" if use_wandb else None,
                run_name="social"+wandb_version if use_wandb else None,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
        social_trainer.train(select_dataset="social",resume_from_checkpoint=resume_from_checkpoint)
        print("have trained social")
        print(model.model.model.layers[31].self_attn.v_proj.mylora_B.social.weight)
    
    elif select_dataset == "agnews":
        wandb.init(project="debug", name="agnews"+wandb_version, reinit=True)
        agnews_output_dir = output_dir + "/agnews"
        agnews_trainer = MyTrainer(
            model=model,
            train_dataset=agnews_train_data, #
            eval_dataset=agnews_val_data, #
            args=transformers.TrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                per_device_eval_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=0,
                num_train_epochs=1,
                learning_rate=learning_rate,
                fp16=True,
                logging_steps=5,
                optim="adamw_hf",
                evaluation_strategy="steps" if val_set_size > 0 else "no",
                save_strategy="steps",
                eval_steps=10 if val_set_size > 0 else None,
                # eval_steps=200 if val_set_size > 0 else None,
                save_steps=40,
                output_dir=agnews_output_dir,
                save_total_limit=1,
                load_best_model_at_end=True if val_set_size > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=group_by_length,
                report_to="wandb" if use_wandb else None,
                run_name="agnews"+wandb_version if use_wandb else None,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
        agnews_trainer.train(select_dataset="agnews",resume_from_checkpoint=resume_from_checkpoint)
        print("have trained agnews")
        print(model.model.model.layers[31].self_attn.v_proj.mylora_B.agnews.weight)
    
    elif select_dataset == "common":
        wandb.init(project="debug", name="common"+wandb_version, reinit=True)
        common_output_dir = output_dir + "/common"
        common_trainer = MyTrainer(
            model=model,
            train_dataset=common_train_data, #
            eval_dataset=common_val_data, #
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
                output_dir=common_output_dir,
                save_total_limit=1,
                load_best_model_at_end=True if val_set_size > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=group_by_length,
                report_to="wandb" if use_wandb else None,
                run_name="common"+wandb_version if use_wandb else None,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
        common_trainer.train(select_dataset="common",resume_from_checkpoint=resume_from_checkpoint)
        print("have trained common")
        print(model.model.model.layers[31].self_attn.v_proj.mylora_B.common.weight)
    
    elif select_dataset == "emotion":
        wandb.init(project="debug", name="emotion"+wandb_version, reinit=True)
        emotion_output_dir = output_dir + "/emotion"
        emotion_trainer = MyTrainer(
            model=model,
            train_dataset=emotion_train_data, #
            eval_dataset=emotion_val_data, #
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
                output_dir=emotion_output_dir,
                save_total_limit=1,
                load_best_model_at_end=True if val_set_size > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=group_by_length,
                report_to="wandb" if use_wandb else None,
                run_name="emotion"+wandb_version if use_wandb else None,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
        emotion_trainer.train(select_dataset="emotion",resume_from_checkpoint=resume_from_checkpoint)
        print("have trained emotion")
        print(model.model.model.layers[31].self_attn.v_proj.mylora_B.emotion.weight)
    
    elif select_dataset == "omw":
        wandb.init(project="debug", name="omw"+wandb_version, reinit=True)
        omw_output_dir = output_dir + "/omw"
        omw_trainer = MyTrainer(
            model=model,
            train_dataset=omw_train_data, #
            eval_dataset=omw_val_data, #
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
                output_dir=omw_output_dir,
                save_total_limit=1,
                load_best_model_at_end=True if val_set_size > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=group_by_length,
                report_to="wandb" if use_wandb else None,
                run_name="omw"+wandb_version if use_wandb else None,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
        omw_trainer.train(select_dataset="omw",resume_from_checkpoint=resume_from_checkpoint)
        print("have trained omw")
        print(model.model.model.layers[31].self_attn.v_proj.mylora_B.omw.weight)
    
    elif select_dataset == "wino":
        wandb.init(project="debug", name="wino"+wandb_version, reinit=True)
        wino_output_dir = output_dir + "/wino"
        wino_trainer = MyTrainer(
            model=model,
            train_dataset=wino_train_data, #
            eval_dataset=wino_val_data, #
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
                output_dir=wino_output_dir,
                save_total_limit=1,
                load_best_model_at_end=True if val_set_size > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=group_by_length,
                report_to="wandb" if use_wandb else None,
                run_name="wino"+wandb_version if use_wandb else None,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
        wino_trainer.train(select_dataset="wino",resume_from_checkpoint=resume_from_checkpoint)
        print("have trained wino")
        print(model.model.model.layers[31].self_attn.v_proj.mylora_B.wino.weight)




    # for name, param in model.named_parameters():
    #     if "lora_P_X" in name:
    #         print(name, "\t", param)
    #     elif "lora_P_H" in name:
    #         print(name, "\t", param)

    print("saving pretrained")
    model.save_pretrained(output_dir)
    print("have saved pretrained\n\n\n\n")

    state_dict = model.state_dict()
    for key, value in state_dict.items():
        print(f"{key}: {value.shape}")
        print(value)
    
    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)

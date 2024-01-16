# lorao

训练阶段：运行 `run_ft.sh`，需要注意的部分如下
- 在 `run_ft.sh` 中可选择 ①传统的单独的 lora `finetune_traditional.py` ②我们的正交 lora `finetune.py`
- 有 4 个数据集 `gsm8k` `boolq` `piqa` `tsa`，选择不同的数据集，需要修改
  1.  `finetune_traditional.py` 或 `finetune.py` 中的 ①数据集的选择、划分 ②trainer 的选择
  2.  `lora.py` 中 `# 🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟` 所在行 tau 值，参考值如下
      | |gsm8k|boolq|piqa|tsa|
      |-|-|-|-|-|
      |tau|1|0|-2|-1.5|



测试阶段：运行 `generate` 开头的 bash 文件，需要注意的部分如下
-  `finetune_traditional.py` 或 `finetune.py` 中的 ①数据集的选择、划分 ②trainer 的选择
-  `lora.py` 中 `# 🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟` 所在行 tau 值
-  手动修改 `lora.py` 中 `self.select_dataset`

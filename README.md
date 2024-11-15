<div align="center">
  <img src="https://github.com/OpenCoder-llm/opencoder-llm.github.io/blob/main/static/images/opencoder_icon.jpg?raw=true" width="30%" alt="OpenCoder-Icon" />
</div>


<p align="center">
    <h1 align="center">
<!--         <img src="https://github.com/user-attachments/assets/93406728-e93f-4a90-9edc-adc346dedbf3"
         alt="Logo" width="65"
        height="65" style="vertical-align: middle;"> -->
        OpenCoder
    </h1>
     <p align="center">⚡ The Open Cookbook for Top-Tier Code Large Language Models ⚡</p>
</p>

<p align="center">
        🏠<a href="https://opencoder-llm.github.io/">Home Page</a>&nbsp&nbsp | &nbsp&nbsp🤗<a href="https://huggingface.co/collections/infly/opencoder-672cec44bbb86c39910fb55e">Model</a>&nbsp&nbsp | &nbsp&nbsp📊<a href="https://huggingface.co/collections/OpenCoder-LLM/opencoder-datasets-672e6db6a0fed24bd69ef1c2">Dataset</a>&nbsp&nbsp | &nbsp&nbsp📄<a href="https://arxiv.org/abs/2411.04905">Paper</a>&nbsp ｜ 🚀<a href="https://huggingface.co/spaces/OpenCoder-LLM/OpenCoder-8B-Instruct">Demo</a>&nbsp&nbsp
</p>

![12](https://github.com/user-attachments/assets/3aa8dd8f-b12a-46e7-a543-d81cfd175d30)

## News
- 🔥🔥 ```2024/11/15``` We have released meta data of **RefineCode** 📊 [RefineCode-code-corpus-meta](https://huggingface.co/datasets/OpenCoder-LLM/RefineCode-raw-code-meta). You can collect your own **RefineCode** referring to this dataset!
- 🔥 ```2024/11/12``` We have released our efficient CodeLLM evaluation framework: [OpenCodeEval](https://github.com/OpenCoder-llm/OpenCoder-llm/tree/main/OpenCodeEval).
- 🔥 ```2024/11/12``` We have released high-quality annealing data 📊 [opc-annealing-corpus](https://huggingface.co/datasets/OpenCoder-LLM/opc-annealing-corpus), which includes algorithmic-corpus along with corresponding synthetic data.
- 🔥 ```2024/11/11``` We have released 55B of recalled pages from [Fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), including 📊 [fineweb-code-corpus](https://huggingface.co/datasets/OpenCoder-LLM/fineweb-code-corpus) and 📊 [fineweb-math-corpus](https://huggingface.co/datasets/OpenCoder-LLM/fineweb-math-corpus).
- 🔥 ```2024/11/09``` We have released 4.5M Post-training data: 📊 [Dataset](https://huggingface.co/collections/OpenCoder-LLM/opencoder-datasets-672e6db6a0fed24bd69ef1c2).
- 🔥 ```2024/11/08``` We have released our models! Please download them from 🤗 [Model](https://huggingface.co/collections/infly/opencoder-672cec44bbb86c39910fb55e).
- 🔥 ```2024/11/07``` We have released our paper on Arxiv: 📄 [OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models](https://arxiv.org/abs/2411.04905).


## Releases
- [ ] Data cleaning pipeline
- [ ] Intermedidate Checkpoints
- [ ] **RefineCode**: Metadata of raw code data 
- [ ] **RefineCode**: Code-related web data
- [x] CodeLLM evaluation framework: OpenCodeEval
- [x] High-quality annealing data
- [x] Post-training data
- [x] Final model weights
- [x] Paper

We are working hard to release all those resources! 💪 


## Introduction

**OpenCoder** is an open and reproducible code LLM family which includes 1.5B and 8B base and chat models, supporting both English and Chinese languages. Starting from scratch, OpenCoder is pretrained on 2.5 trillion tokens composed of 90% raw code and 10% code-related web data, and supervised finetuned on over 4.5M high-quality SFT examples, finally reaching the performance of top-tier code LLMs. We provide not only model weights and inference code, but also the reproducible training data, the complete data processing pipeline, rigorous experimental ablation results, and detailed training protocols. Empowering researchers to build and innovate, OpenCoder is your open foundation for advancing code AI. 

- **Complete Open Source**: OpenCoder ensures full transparency by releasing not only the model weights and forthcoming inference code but also the complete data-cleaning code for training. This release includes high-quality synthetic data, an extensive set of checkpoints, and a dataset of over 4.5 million supervised fine-tuning (SFT) entries, making OpenCoder one of the most comprehensively open-sourced models available.
- **Comprehensive Experimental Analysis**: OpenCoder is rigorously tested through extensive ablation studies on various data-cleaning strategies and training processes, including file-level and repository-level deduplication experiments, ensuring thorough exploration and validation of the model’s performance.
- **High-Quality Synthetic Data**: OpenCoder provides a fully developed synthetic data generation process and over 4.5 million SFT data entries, establishing a robust data foundation for model training and evaluation.
- **Exceptional Performance**: OpenCoder achieves high performance across multiple language model benchmarks, positioning it among the leading open-source models for code.


## Models

<!-- |         Model         | Sequence Length |                                Download                                 |
|:---------------------:|:---------------:|:-----------------------------------------------------------------------:|
| OpenCoder-1.5B-Base  |      4K       | 🤗 [HuggingFace](https://huggingface.co/infly/OpenCoder-1.5B-Base)  |
| OpenCoder-8B-Base  |      8K       | 🤗 [HuggingFace](https://huggingface.co/infly/OpenCoder-8B-Base)  |
| OpenCoder-1.5B-Instruct  |      4K       | 🤗 [HuggingFace](https://huggingface.co/infly/OpenCoder-1.5B-Instruct) |
| OpenCoder-8B-Instruct  |      8K       | 🤗 [HuggingFace](https://huggingface.co/infly/OpenCoder-8B-Instruct) | -->

|         Model         | Sequence Length |                   HuggingFace                 |      wisemodel    |        
|:---------------------:|:---------------:|:-----------------------------------------------------------------------:|:------------------------------------------|
| OpenCoder-1.5B-Base  |      4K       | [🤗HuggingFace](https://huggingface.co/infly/OpenCoder-1.5B-Base)  |  [<img src="https://github.com/OpenCoder-llm/opencoder-llm.github.io/blob/main/static/images/wisemodel_logo.png?raw=true" height="12">](https://wisemodel.cn/models/OpenCoder/OpenCoder-1.5B-Base) |
| OpenCoder-8B-Base  |      8K       | [🤗HuggingFace](https://huggingface.co/infly/OpenCoder-8B-Base)  | [<img src="https://github.com/OpenCoder-llm/opencoder-llm.github.io/blob/main/static/images/wisemodel_logo.png?raw=true" height="12">](https://wisemodel.cn/models/OpenCoder/OpenCoder-8B-Base) |
| OpenCoder-1.5B-Instruct  |      4K       | [🤗HuggingFace](https://huggingface.co/infly/OpenCoder-1.5B-Instruct) | [<img src="https://github.com/OpenCoder-llm/opencoder-llm.github.io/blob/main/static/images/wisemodel_logo.png?raw=true" height="12">](https://wisemodel.cn/models/OpenCoder/OpenCoder-1.5B-Instruct) |
| OpenCoder-8B-Instruct  |      8K       | [🤗HuggingFace](https://huggingface.co/infly/OpenCoder-8B-Instruct) | [<img src="https://github.com/OpenCoder-llm/opencoder-llm.github.io/blob/main/static/images/wisemodel_logo.png?raw=true" height="12">](https://wisemodel.cn/models/OpenCoder/OpenCoder-8B-Instruct) |


## Datasets

### Pre-training
|         Dataset       | Size |                                Download                                 |
|:---------------------:|:---------------:|:-----------------------------------------------------------------------:|
| fineweb-code-corpus  |      148 GB       | [🤗HuggingFace](https://huggingface.co/datasets/OpenCoder-LLM/fineweb-code-corpus)  |
| fineweb-math-corpus  |       10 GB    | [🤗HuggingFace](https://huggingface.co/datasets/OpenCoder-LLM/fineweb-math-corpus)  |
| opc-annealing-corpus  |      24 GB    | [🤗HuggingFace](https://huggingface.co/datasets/OpenCoder-LLM/opc-annealing-corpus)  |


### Post-training

|         Dataset       | Num |                                Download                                 |
|:---------------------:|:---------------:|:-----------------------------------------------------------------------:|
| opc-sft-stage1  |      4.21 M       | [🤗HuggingFace](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage1)  |
| opc-sft-stage2  |      375 K      | [🤗HuggingFace](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2)  |


**This is not the end; we are organizing the remaining data and uploading it progressively.**

## Performance
<img src="https://github.com/user-attachments/assets/7f5a49b2-9539-4185-91fa-fd32c1315b2a" width="75%">
<img src="https://github.com/user-attachments/assets/81c6e686-0ed0-4eb5-8fb8-a651750ec346" width="75%">

## Get Started
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "infly/OpenCoder-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto",
                                             trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

messages=[
    { 'role': 'user', 'content': "write a quick sort algorithm in python."}
]

inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

outputs = model.generate(inputs, max_new_tokens=512, do_sample=False)

result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
print(result)
```

## Citation
If you find our work helpful, feel free to give us a cite :-)

```bibtex
@inproceedings{Huang2024OpenCoderTO,
  title={OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models},
  author={Siming Huang and Tianhao Cheng and Jason Klein Liu and Jiaran Hao and Liuyihan Song and Yang Xu and J. Yang and J. H. Liu and Chenchen Zhang and Linzheng Chai and Ruifeng Yuan and Zhaoxiang Zhang and Jie Fu and Qian Liu and Ge Zhang and Zili Wang and Yuan Qi and Yinghui Xu and Wei Chu},
  year={2024},
  url={https://arxiv.org/pdf/2411.04905}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=OpenCoder-llm/OpenCoder-llm&type=Date)](https://star-history.com/#OpenCoder-llm/OpenCoder-llm&Date)

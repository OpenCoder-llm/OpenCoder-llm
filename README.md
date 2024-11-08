<p align="center">
    <h1 align="center">
        <img src="https://github.com/user-attachments/assets/93406728-e93f-4a90-9edc-adc346dedbf3"
         alt="Logo" width="65"
        height="65" style="vertical-align: middle;">
        OpenCoder
    </h1>
     <p align="center">⚡ The Cookbook for Top-Tier Open CodeLLM ⚡</p>
</p>

<!-- 👉 HomePage -->

![12](https://github.com/user-attachments/assets/3aa8dd8f-b12a-46e7-a543-d81cfd175d30)

## Introduction

**OpenCoder** is an open and reproducible code LLM family which includes 1.5B and 8B base and chat models, supporting both English and Chinese languages. Starting from scratch, OpenCoder is pretrained on 2.5 trillion tokens composed of 90% raw code and 10% code-related web data, and supervised finetuned on over 450M high-quality SFT examples, finally reaching the performance of top-tier code LLMs. We provide not only model weights and inference code, but also the reproducible training data, the complete data processing pipeline, rigorous experimental ablation results, and detailed training protocols. Empowering researchers to build and innovate, OpenCoder is your open foundation for advancing code AI. 

- **Complete Open Source**: OpenCoder ensures full transparency by releasing not only the model weights and forthcoming inference code but also the complete data-cleaning code for training. This release includes high-quality synthetic data, an extensive set of checkpoints, and a dataset of over 4.5 million supervised fine-tuning (SFT) entries, making OpenCoder one of the most comprehensively open-sourced models available.
- **Comprehensive Experimental Analysis**: OpenCoder is rigorously tested through extensive ablation studies on various data-cleaning strategies and training processes, including file-level and repository-level deduplication experiments, ensuring thorough exploration and validation of the model’s performance.
- **High-Quality Synthetic Data**: OpenCoder provides a fully developed synthetic data generation process and over 4.5 million SFT data entries, establishing a robust data foundation for model training and evaluation.
- **Exceptional Performance**: OpenCoder achieves high performance across multiple language model benchmarks, positioning it among the leading open-source models for code.


## Models

|         Model         | Sequence Length |                                Download                                 |
|:---------------------:|:---------------:|:-----------------------------------------------------------------------:|
| OpenCoder-1.5B-Base  |      4K       | 🤗 [HuggingFace](https://huggingface.co/infly/OpenCoder-1.5B-Base)  |
| OpenCoder-8B-Base  |      8K       | 🤗 [HuggingFace](https://huggingface.co/infly/OpenCoder-8B-Base)  |
| OpenCoder-1.5B-Instruct  |      4K       | 🤗 [HuggingFace](https://huggingface.co/infly/OpenCoder-1.5B-Instruct) |
| OpenCoder-8B-Instruct  |      8K       | 🤗 [HuggingFace](https://huggingface.co/infly/OpenCoder-8B-Instruct) |

## Datasets

|         Dataset       | Num |                                Download                                 |
|:---------------------:|:---------------:|:-----------------------------------------------------------------------:|
| OpenCoder-SFT-Stage1  |      4.21 M       | 🤗 [HuggingFace](https://huggingface.co/datasets/OpenCoder-LLM/opencoder-sft-stage1)  |
| OpenCoder-SFT-Stage2  |      375 K      | 🤗 [HuggingFace](https://huggingface.co/datasets/OpenCoder-LLM/opencoder-sft-stage2)  |

## Performance

<!-- ![benchmark_base](https://github.com/user-attachments/assets/7f5a49b2-9539-4185-91fa-fd32c1315b2a) -->
<!-- ![benchmark_instruct](https://github.com/user-attachments/assets/81c6e686-0ed0-4eb5-8fb8-a651750ec346) -->
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
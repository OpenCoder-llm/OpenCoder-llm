#!/bin/bash

# 定义要下载的文件的 URL 列表
urls=(
    "https://github.com/bigcode-project/bigcodebench-annotation/releases/download/v0.1.1/BigCodeBench-Hard.jsonl.gz"
    "https://github.com/bigcode-project/bigcodebench-annotation/releases/download/v0.1.1/BigCodeBench.jsonl.gz"
    "https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz"
    "https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz"
    "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
)

# 下载并解压每个文件
for url in "${urls[@]}"; do
    # 获取文件名
    filename=$(basename "$url")

    # 删除已有的压缩文件和解压后的文件，确保不会重复
    [ -f "$filename" ] && rm "$filename"
    [ -f "${filename%.gz}" ] && rm "${filename%.gz}"

    echo "Downloading $url..."
    wget "$url"

    echo "Unzipping $filename..."
    gunzip "$filename"
done

echo "All files have been downloaded and unzipped."

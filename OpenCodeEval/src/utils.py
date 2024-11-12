import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([os.path.dirname(ROOT), os.path.dirname(os.path.dirname(ROOT))])
import re
import gzip
import json
import itertools
import numpy as np

from typing import Dict, List, Union, Iterable
from collections import defaultdict
from transformers import AutoTokenizer

python_pattern = r"```python[ \t]*[\r\n]+(.*?)[ \t]*[\r\n]+```"
python_re = re.compile(python_pattern, re.DOTALL | re.IGNORECASE)

def python_extract(text: str) -> str:
    match = python_re.search(text)
    if match:
        return match.group(1)
    else:
        return ""

def refine_text(text: str) -> str:
    text =  text.replace("\t", "    ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip() + "\n"

def format_test_example(q, tests, code: str=None):
    prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(q.strip(), "\n".join(tests))
    if code:
        code = code.replace("\r", "").replace("\t", "    ")
        prompt += "\n>>> Code:\n```python\n{}\n```".format(code)
    return prompt

def make_chat_prompt(prompt: str,
                     tokenizer: AutoTokenizer,
                     response_prefix: str = ""
                    ) -> str:
    # directly return prompt if it does not have a tokenizer.chat_template
    if tokenizer.chat_template:

        if 'ckpt' in tokenizer.name_or_path or 'checkpoint' in tokenizer.name_or_path or 'ckp' in tokenizer.name_or_path:

            return '''
### Instruction:
{}
### Response:
'''.format(prompt.strip()).lstrip()
        
        else:
            prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content":  prompt},
            ],
            tokenize = False,
            add_generation_prompt = True
        ) + response_prefix
        
    return prompt[len(tokenizer.bos_token):] if prompt.startswith(tokenizer.bos_token) else prompt


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r", encoding="utf-8") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))

def group_and_count(lst, group_key, count_key):

    grouped_counts = defaultdict(int)
    
    for item in lst:
        group = item.get(group_key)
        if group not in grouped_counts:
            grouped_counts[group] = 0
        if item.get(count_key) == True:
            grouped_counts[group] += 1
    
    return list(grouped_counts.values())

def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

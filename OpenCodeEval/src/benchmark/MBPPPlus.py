import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([os.path.dirname(ROOT), os.path.dirname(os.path.dirname(ROOT))])

from concurrent.futures import ThreadPoolExecutor, as_completed

from benchmark.base import Benchmark
from eval.execution import check_correctness
from sanitize import sanitize
from tqdm import tqdm
from utils import refine_text, stream_jsonl


class MBPPPlus(Benchmark):

    name: str = "MBPPPlus"
    path: str = os.path.abspath(os.path.join(ROOT, "../data/MBPPPlus.jsonl"))

    general_stop_words = [
        "<|endoftext|>",
        "<|endofmask|>",
        "</s>",
        "\nif __name__",
        "\ndef main(",
        "\nprint(",
        "\n```\n",
    ]

    completion_stop_words = ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert "]

    imports = [
        "import math",
        "import re",
        "import sys",
        "import copy",
        "import datetime",
        "import itertools",
        "import collections",
        "import heapq",
        "import functools",
        "import hashlib",
        "import numpy",
        "import numpy as np",
        "import string",
        "from typing import *",
        "from collections import *",
    ]

    def __init__(
        self,
        name: str = "MBPPPlus",
        num_samples: int = 1,
        num_workers: int = 16,
        timeout: float = 3.0,
        prompt_type: str = "Instruction",
    ):

        super().__init__()
        self.name = name
        self.num_samples = num_samples
        self.num_workers = num_workers
        self.timeout = timeout
        self.prompt_type = prompt_type

    def get_task(self):

        return list(stream_jsonl(filename=self.path))

    def format_prompt(
        self,
        promblem: str,
        test: str,
    ) -> str:
        # promblem = f"You are an expert Python programmer, and here is your task:\n{promblem}\n"
        # test = f"Your code should pass the test:\n{test}\n"
        # prompt = promblem + test
        prompt = promblem + "\n" + test
        return prompt

    def get_prompt(self):

        assert self.prompt_type == "Instruction", f"Prompt type must be Instruction for {self.name}"

        task_set = self.get_task()
        prompts = []
        for task_data in task_set:
            prompt = self.format_prompt(task_data["text"], task_data["test_list"][0])
            prompts.append(refine_text(prompt))

        return prompts

    def postprocess_generation(self, generation_group):

        solution_group = []
        for generation_samples in generation_group:
            solution_group.append([sanitize(generation) for generation in generation_samples])

        return solution_group

    def process_results(self, solution_group):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """

        task_set = self.get_task()

        evals = []
        for index, task_data in enumerate(task_set):
            task_id = task_data["task_id"]
            solutions_list = solution_group[index]
            assert (
                len(solutions_list) == self.num_samples
            ), f"Num completions : {len(solutions_list)} not match Num samples: {self.num_samples}"
            for solution_id, solution_data in enumerate(solutions_list):

                if self.name == "MBPPPlus":
                    test_code = task_data["test"]
                elif self.name == "MBPPBase":
                    test_code = "\n".join(task_data["test_imports"]) + "\n\n" + "\n".join(task_data["test_list"])
                else:
                    raise ValueError(f"Invalid benchmark name: {self.name}")

                solution = (
                    "\n".join(self.imports)
                    + "\n\n"
                    + solution_data
                    + "\n\n"
                    + "\n".join(task_data["test_imports"])
                    + "\n\n"
                    + test_code
                    + "\n\n"
                )
                evals.append({"task_id": task_id, "solution_id": solution_id, "solution": solution})

        print(evals[0]["solution"])

        with ThreadPoolExecutor(self.num_workers) as executor:
            futures = []
            for eval in evals:
                args = (eval["task_id"], eval["solution_id"], eval["solution"], self.timeout)
                future = executor.submit(check_correctness, *args)
                futures.append(future)

            evaluations_set = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Completing tasks"):
                result = future.result()
                evaluations_set.append(result)

        evaluations_set = sorted(evaluations_set, key=lambda x: (x["task_id"], x["solution_id"]))

        return evaluations_set

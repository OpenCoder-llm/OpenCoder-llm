import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([os.path.dirname(ROOT), os.path.dirname(os.path.dirname(ROOT))])

from typing import Literal

from benchmark.base import Benchmark
from eval.unit_test import check_correctness
from sanitize import sanitize
from utils import refine_text, stream_jsonl


class BigCodeBench(Benchmark):

    name: str = "BigCodeBench"

    fullset_path = os.path.abspath(os.path.join(ROOT, "../data/BigCodeBench.jsonl"))
    subset_path = os.path.abspath(os.path.join(ROOT, "../data/BigCodeBench_Hard.jsonl"))

    def __init__(
        self,
        name: str = "BigCodeBench",
        timeout: float = 10.0,
        prompt_type: Literal["Completion", "Instruction"] = "Completion",
    ):

        super().__init__()

        self.name = name
        self.timeout = timeout
        self.prompt_type = prompt_type

        if self.name == "BigCodeHard":
            self.path = self.subset_path
        elif self.name == "BigCodeBench":
            self.path = self.fullset_path

        self.tasks = self.get_task()

    def get_task(self):
        """
        Get the task data from the jsonl file into a dictionary.
        """

        tasks = {}

        for task_data in stream_jsonl(filename=self.path):

            task_id = int(task_data["task_id"].split("/")[-1])

            tasks[task_id] = task_data

        return tasks

    def get_prompt(self):
        """
        Builds the prompt for the LM to generate from.
        """

        prompts = []
        for task_id, task_data in self.tasks.items():

            if self.prompt_type == "Completion":
                prompt = task_data["complete_prompt"]
            elif self.prompt_type == "Instruction":
                prompt = task_data["instruct_prompt"]

            prompts.append(dict(task_id=task_id, prompt=refine_text(prompt)))

        return prompts

    def postprocess_generation(self, generation):
        """
        Postprocess the generations.
        """

        entry_point = self.tasks[generation["task_id"]]["entry_point"]

        result = dict(
            task_id=generation["task_id"],
            completion_id=generation["completion_id"],
            solution=sanitize(generation["completion"], entry_point),
        )

        return result

    def process_results(self, solution):
        """
        Takes the list of LM generations and evaluates them against the test cases
        """

        task_data = self.tasks[solution["task_id"]]

        code = task_data["code_prompt"] + "\n" + "    pass\n" + "\n" + solution["solution"] + "\n"

        result = check_correctness(
            solution["task_id"], solution["completion_id"], code, task_data["test"], self.timeout
        )

        return result

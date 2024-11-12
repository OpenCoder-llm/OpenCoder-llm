import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([os.path.dirname(ROOT), os.path.dirname(os.path.dirname(ROOT))])

from benchmark.base import Benchmark
from sanitize import sanitize
from eval.execution import check_correctness
from utils import refine_text, stream_jsonl

class HumanEval(Benchmark):

    name: str = "HumanEval"
    
    base_path: str = os.path.abspath(os.path.join(ROOT, "../data/HumanEval.jsonl"))
    plus_path: str = os.path.abspath(os.path.join(ROOT, "../data/HumanEvalPlus.jsonl"))

    def __init__(self,
                 name: str = "HumanEval",
                 timeout: float = 3.0,
                 prompt_type: str = "Completion"): 
        super().__init__()
        
        self.name = name
        self.timeout = timeout
        self.prompt_type = prompt_type

        if self.name == "HumanEvalPlus":
            self.path = self.plus_path
        elif self.name == "HumanEval":
            self.path = self.base_path

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

        assert self.prompt_type == "Completion", f"Prompt type must be Completion for HumanEval"

        prompts = []
        for task_id, task_data in self.tasks.items():
            prompts.append(
                dict(
                    task_id = task_id,
                    prompt = refine_text(task_data['prompt'])
                )
            )
        return prompts

    def postprocess_generation(self, generation):
        """
        Postprocess the generations.
        """

        entry_point = self.tasks[generation['task_id']]["entry_point"]

        result = dict(
            task_id = generation['task_id'],
            completion_id = generation['completion_id'],
            solution = sanitize(generation['completion'], entry_point)
        )

        return result

    def process_results(self, solution):
        """
        Takes the list of LM generations and evaluates them against the test cases
        """

        task_data = self.tasks[solution['task_id']]

        code = ("\n".join(self.imports) + "\n"
                    + task_data["prompt"] + "\n"
                    + "    pass\n" + "\n"
                    + solution['solution'] + "\n"
                    + task_data['test'] + "\n"
                    + f"check({task_data['entry_point']})"
                )
        
        result = check_correctness(solution['task_id'],
                                   solution['completion_id'],
                                   code,
                                   self.timeout)
        
        return result
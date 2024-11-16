import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List

import numpy as np
from args import check_args, get_args
from backend.vllm import VllmGenerator
from factory import BenchmarkFactory
from tqdm import tqdm
from utils import estimate_pass_at_k, group_and_count, refine_text, write_jsonl


def multi_process_function(function: Callable, parameters: List, num_workers: int = 1, desc: str = "Completing tasks"):

    if num_workers > len(parameters) or num_workers > os.cpu_count():
        num_workers = min(os.cpu_count(), len(parameters))

    with ThreadPoolExecutor(num_workers) as executor:
        futures = []
        for param in parameters:
            future = executor.submit(function, param)
            futures.append(future)

        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            result = future.result()
            results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser()
    args = get_args(parser)
    args = check_args(args)

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    task = BenchmarkFactory.get_task(args)

    decoder = VllmGenerator(
        model_name=args.model_name,
        model_type=args.model_type,
        tokenizer_name=args.tokenizer_name,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        temperature=args.temperature,
        trust_remote_code=args.trust_remote_code,
        max_tokens=args.max_tokens,
    )

    prompts = task.get_prompt()

    for prompt in prompts:
        prompt["prompt"] = refine_text(args.prompt_prefix + prompt["prompt"] + args.prompt_suffix)
    write_jsonl(save_path + "/prompts.jsonl", prompts)

    end_words = (
        task.general_stop_words + task.completion_stop_words if args.model_type == "Base" else task.general_stop_words
    )
    generations = decoder.generate(prompts, args.num_samples, end_words, args.response_prefix, args.response_suffix)
    write_jsonl(save_path + "/generations.jsonl", generations)

    solutions = multi_process_function(
        function=task.postprocess_generation,
        parameters=generations,
        num_workers=args.num_workers,
        desc="Post-processing solutions",
    )
    write_jsonl(save_path + "/solutions.jsonl", solutions)

    evaluations = multi_process_function(
        function=task.process_results, parameters=solutions, num_workers=args.num_workers, desc="Evaluating solutions"
    )
    write_jsonl(save_path + "/evaluation.jsonl", evaluations)

    result_list = group_and_count(evaluations, group_key="task_id", count_key="passed")
    pass_rate = float(np.mean(estimate_pass_at_k(num_samples=args.num_samples, num_correct=result_list, k=1)))
    write_jsonl(save_path + "/result.json", [{"score": pass_rate}])
    print("Pass@1:", pass_rate)


if __name__ == "__main__":
    main()

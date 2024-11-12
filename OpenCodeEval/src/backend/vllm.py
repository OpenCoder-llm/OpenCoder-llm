import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([os.path.dirname(ROOT), os.path.dirname(os.path.dirname(ROOT))])

import gc
import torch
from loguru import logger

from tqdm import tqdm
from typing import List, Dict

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

from backend.base import Generator

from utils import make_chat_prompt, refine_text

class VllmGenerator(Generator):
    def __init__(self,
                 model_name: str,
                 tokenizer_name: str = None,
                 model_type: str = "Instruction",
                 dtype: str = "bfloat16",
                 batch_size : int = 1,
                 temperature : float = 0.0,
                 max_tokens : int = 1024,
                 num_gpus: int = 1,
                 trust_remote_code: bool = True,
                ) -> None:
        super().__init__(model_name)

        print("Initializing a decoder model: {} ...".format(model_name))
        self.tokenizer_name = tokenizer_name if tokenizer_name is not None else model_name
        self.model_type = model_type
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.dtype = dtype
        self.num_gpus = num_gpus
        self.trust_remote_code = trust_remote_code
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name,
                                                       model_max_length = self.max_tokens,
                                                       trust_remote_code = self.trust_remote_code,
                                                       use_fast = True)

        self.model = LLM(model = self.model_name,
                         tokenizer = None,
                         max_model_len = self.max_tokens,
                         tensor_parallel_size = self.num_gpus,
                         trust_remote_code = self.trust_remote_code)
        
        self.model.set_tokenizer(tokenizer = self.tokenizer)

    def is_chat(self) -> bool:
        if self.model_type == "Chat":
            assert self.tokenizer.chat_template is not None
            return True
        else:
            return False
        
    def release_memory(self):
        destroy_model_parallel
        destroy_distributed_environment()
        del self.model.llm_engine.model_executor
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def generate(self,
                 prompt_set: List[Dict],
                 num_samples: int = 200,
                 eos: List[str] = None,
                 response_prefix: str = "",
                 response_suffix: str = ""
                ) -> List[str]:

        if self.is_chat():
            for prompt in prompt_set:
                prompt['prompt'] = make_chat_prompt(prompt['prompt'], self.tokenizer, response_prefix)

        logger.info("Example Prompt:\n{}", prompt_set[0]['prompt'])

        sampling_params = SamplingParams(
            n = num_samples,
            temperature = self.temperature,
            max_tokens = self.max_tokens,
            top_p = 1.0,
            stop = eos,
        )

        generation_set = []

        for batch_start in tqdm(range(0, len(prompt_set), self.batch_size)):
            batch_prompt = prompt_set[batch_start : batch_start + self.batch_size]
            batch_outputs = self.model.generate(
                [prompt['prompt'] for prompt in batch_prompt],
                sampling_params,
                use_tqdm = False,
            )

            for prompt, output in zip(batch_prompt, batch_outputs):

                completions = [completion.text for completion in output.outputs]

                if not self.is_chat():
                    completions = [refine_text(prompt["prompt"] + "\n\n" + response_prefix + sample + response_suffix) for sample in completions]
                else:
                    completions = [refine_text(response_prefix + sample + response_suffix) for sample in completions]

                for completion_id, completion in enumerate(completions):
                    generation_set.append({
                        'task_id': prompt['task_id'],
                        'completion_id': completion_id,
                        'completion': completion
                    })

        assert len(generation_set) == len(prompt_set) * num_samples, "Number of generations does not match the expected number."

        self.release_memory()

        return generation_set
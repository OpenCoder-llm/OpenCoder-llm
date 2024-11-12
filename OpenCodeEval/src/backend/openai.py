import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([os.path.dirname(ROOT), os.path.dirname(os.path.dirname(ROOT))])

import openai

from tqdm import tqdm
from typing import List
from openai import OpenAI
from vllm import LLM, SamplingParams


from backend.base import Generator

from utils import make_chat_prompt, refine_text

class OpenaiGenerator(Generator):
    def __init__(self,
                 model_name: str,
                 model_type: str = "Instruction",
                 batch_size : int = 1,
                 temperature : float = 0.0,
                 max_tokens : int = 1024,
                 eos: List[str] = None,
                 num_gpus: int = 1,
                 trust_remote_code: bool = True,
                ) -> None:
        super().__init__(model_name)

        print("Initializing a decoder model: {} ...".format(model_name))
        self.model_type = model_type
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_gpus = num_gpus
        self.trust_remote_code = trust_remote_code
        self.eos = eos
        

        self.model = LLM(model = self.model_name,
                         max_model_len = 2048,
                         tensor_parallel_size = self.num_gpus,
                         trust_remote_code = self.trust_remote_code)
        
        self.model.set_tokenizer(tokenizer=self.tokenizer)

    def is_chat(self) -> bool:
        if self.model_type == "Chat":
            assert self.tokenizer.chat_template is not None
            return True
        else:
            return False

    def generate(self, prompts: List[str],
                 num_samples: int = 200,
                 response_prefix: str = ""
                ) -> List[str]:

        if self.is_chat():
            prompts = [make_chat_prompt(prompt, self.tokenizer, response_prefix) for prompt in prompts]
        print(prompts[0])
        sample_prompts = [prompt for prompt in prompts for _ in range(num_samples)]

        assert len(sample_prompts) == (len(prompts) * num_samples)
        assert all(sample_prompts[i:i + num_samples] == [sample_prompts[i]] * num_samples for i in range(0, len(sample_prompts), num_samples))

        generations = []

        for batch_start in tqdm(range(0, len(sample_prompts), self.batch_size)):
            batch = sample_prompts[batch_start : batch_start + self.batch_size]
            batch_outputs = self.model.generate(
                batch,
                SamplingParams(
                    temperature = self.temperature,
                    max_tokens = 768,
                    top_p = 1.0,
                    stop = self.eos,
                ),
                use_tqdm = False,
            )

            batch_generations = []
            for output in batch_outputs:
                prompt = output.prompt
                generation = output.outputs[0].text
                if not self.is_chat():
                    batch_generations.append(refine_text(prompt + "\n" + generation))
                else:
                    batch_generations.append(refine_text(generation))

            generations.extend(batch_generations)

        grouped_generatuons = [generations[i:i + num_samples] for i in range(0, len(generations), num_samples)]
        assert len(grouped_generatuons) == len(prompts)
        assert all(len(grouped_generatuons[i]) == num_samples for i in range(len(grouped_generatuons)))

        return grouped_generatuons
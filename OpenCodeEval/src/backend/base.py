import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([os.path.dirname(ROOT), os.path.dirname(os.path.dirname(ROOT))])

from abc import ABC, abstractmethod


class Generator(ABC):

    model_name: str = None

    def __init__(self, model_name: str) -> None:
        """
        :param stop_words: list
            list of stop words if the generation uses a stopping criteria during generation
        :param requires_execution: bool
            wheter the task requires code execution during evaluation or not
        """
        self.model_name = model_name

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""

    @abstractmethod
    def generate(self):
        """Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        """

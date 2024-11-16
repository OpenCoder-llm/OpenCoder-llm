import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([os.path.dirname(ROOT), os.path.dirname(os.path.dirname(ROOT))])

from benchmark.BigCodeBench import BigCodeBench
from benchmark.HumanEval import HumanEval
from benchmark.LeetCode import LeetCode
from benchmark.MBPP import MBPP
from benchmark.MBPPPlus import MBPPPlus


class BenchmarkFactory:
    @staticmethod
    def get_task(args):
        if args.task == "HumanEval" or args.task == "HumanEvalPlus":
            return HumanEval(name=args.task, prompt_type=args.prompt_type)
        elif args.task == "MBPP":
            return MBPP(name=args.task, prompt_type=args.prompt_type)
        elif args.task == "MBPPPlus" or args.task == "MBPPBase":
            return MBPPPlus(name=args.task, prompt_type=args.prompt_type)
        elif args.task == "LeetCode":
            return LeetCode(name=args.task, prompt_type=args.prompt_type)
        elif args.task == "BigCodeBench" or args.task == "BigCodeHard":
            return BigCodeBench(name=args.task, prompt_type=args.prompt_type)
        else:
            raise ValueError("Unknown Task type")

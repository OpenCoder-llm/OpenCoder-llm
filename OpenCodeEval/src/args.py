import os

from loguru import logger

ROOT = os.path.dirname(os.path.abspath(__file__))


def get_args(parser):

    # ===================Model Parser===================
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--tokenizer_name", default=None, type=str)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--backend", default="vllm", type=str, choices=["vllm", "hf", "openai"])
    parser.add_argument(
        "--task",
        default="HumanEval",
        type=str,
        choices=[
            "HumanEval",
            "MBPP",
            "LeetCode",
            "BigCodeHard",
            "BigCodeBench",
            "HumanEvalPlus",
            "MBPPPlus",
            "MBPPBase",
        ],
    )
    parser.add_argument("--prompt_type", default="Instruction", type=str, choices=["Completion", "Instruction"])
    parser.add_argument("--model_type", default="Chat", type=str, choices=["Base", "Chat"])

    # ===================Computer Parser===================
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--num_workers", default=1, type=int)

    parser.add_argument("--save_path", default="save", type=str)
    parser.add_argument("--batch_size", default=164, type=int)
    parser.add_argument("--num_samples", default=1, type=int)
    parser.add_argument("--max_tokens", default=2048, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)

    parser.add_argument("--prompt_prefix", default="", type=str)
    parser.add_argument("--prompt_suffix", default="", type=str)
    parser.add_argument("--response_prefix", default="", type=str)
    parser.add_argument("--response_suffix", default="", type=str)

    return parser.parse_args()


def check_args(args):

    if args.tokenizer_name is None:
        args.tokenizer_name = args.model_name

    # When eval the pretrained model, it can not response to the instrcution, so suggest using the Completion prompt
    if args.model_type == "Base" and args.prompt_type == "Instruction":
        logger.warning("Prompt type must be Completion for Base Model")

    if args.num_samples > 1 and args.temperature != 0.0:
        logger.error("Temperature is not allowed when num_samples > 1")

    # When eval the chat model, it can not reply to the uncompleted code, so suggest using the Instruction prompt
    if args.model_type == "Chat" and args.prompt_type == "Completion":
        if args.prompt_prefix == "":
            logger.warning("Prompt prefix is not set, using default prefix")
            args.prompt_prefix = (
                "Please provide a self-contained Python script that "
                "solves the following problem in a markdown code block:\n```python\n"
            )
        if args.prompt_suffix == "":
            logger.warning("Prompt suffix is not set, using default suffix")
            args.prompt_suffix = "\n```\n"

    return args

from dataclasses import dataclass
from typing import Optional

import torch

from src.generation.load_model import load_model
from src.generation.prompts import build_prompt


@dataclass
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 2048


def generate_trajectory(
    problem: str,
    config: Optional[GenerationConfig] = None,
):
    """
    Generate one reasoning trajectory for a problem.
    """

    if config is None:
        config = GenerationConfig()

    bundle = load_model()

    prompt = build_prompt(problem)

    inputs = bundle.tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(bundle.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output = bundle.model.generate(
            **inputs,
            do_sample=True,
            temperature=config.temperature,
            top_p=config.top_p,
            max_new_tokens=config.max_new_tokens,
        )

    generated_tokens = output[0][inputs["input_ids"].shape[1]:]

    trajectory = bundle.tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return trajectory



if __name__ == "__main__":
    test_problem = "What is 2 + 2?"

    trajectory = generate_trajectory(test_problem)

    print("\nGenerated trajectory:")
    print(trajectory)
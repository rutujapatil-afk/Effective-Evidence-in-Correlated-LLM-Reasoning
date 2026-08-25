from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"


@dataclass
class ModelBundle:
    tokenizer: object
    model: object
    device: str


def get_device() -> str:
    """
    Select the inference device.

    CUDA will be used automatically when an NVIDIA GPU
    is available. Otherwise, CPU is used.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name: str = MODEL_NAME) -> ModelBundle:
    device = get_device()

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading model: {model_name}")
    print(f"Device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    model.to(device)
    model.eval()

    print("Model loaded successfully.")

    return ModelBundle(
        tokenizer=tokenizer,
        model=model,
        device=device,
    )


if __name__ == "__main__":
    bundle = load_model()

    print("\nModel verification:")
    print(f"Device: {bundle.device}")
    print(f"Model type: {type(bundle.model).__name__}")
    print(f"Tokenizer type: {type(bundle.tokenizer).__name__}")
import re
from fractions import Fraction
from typing import Optional


def normalize_answer(answer: str) -> str:
    """
    Normalize a mathematical answer for deterministic comparison.
    """

    answer = answer.strip()

    # Remove common LaTeX delimiters.
    answer = answer.replace("$", "")
    answer = answer.replace("\\(", "")
    answer = answer.replace("\\)", "")

    # Normalize whitespace.
    answer = re.sub(r"\s+", " ", answer)

    # Remove trailing punctuation.
    answer = answer.rstrip(".,;:")

    return answer.strip()


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the content of the final \\boxed{...} expression.

    Handles nested braces using a small parser rather than
    a naive regular expression.
    """

    matches = list(re.finditer(r"\\boxed\s*\{", text))

    if not matches:
        return None

    start = matches[-1].end()
    depth = 1

    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1

            if depth == 0:
                return text[start:i]

    return None


def normalize_numeric(value: str) -> Optional[Fraction]:
    """
    Convert simple integer, decimal, or fraction answers
    into an exact Fraction where possible.
    """

    value = normalize_answer(value)

    # Remove LaTeX fraction syntax.
    fraction_match = re.fullmatch(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        value,
    )

    if fraction_match:
        numerator = fraction_match.group(1).strip()
        denominator = fraction_match.group(2).strip()

        try:
            return Fraction(
                int(numerator),
                int(denominator),
            )
        except ValueError:
            return None

    # Standard fraction.
    if re.fullmatch(r"-?\d+\s*/\s*-?\d+", value):
        numerator, denominator = value.split("/")

        try:
            return Fraction(
                int(numerator.strip()),
                int(denominator.strip()),
            )
        except ValueError:
            return None

    # Integer.
    if re.fullmatch(r"-?\d+", value):
        return Fraction(int(value), 1)

    # Decimal.
    if re.fullmatch(r"-?\d+\.\d+", value):
        try:
            return Fraction(value)
        except ValueError:
            return None

    return None


def answers_match(predicted: str, gold: str) -> bool:
    """
    Compare predicted and gold answers deterministically.
    """

    predicted_normalized = normalize_answer(predicted)
    gold_normalized = normalize_answer(gold)

    if predicted_normalized == gold_normalized:
        return True

    predicted_numeric = normalize_numeric(predicted_normalized)
    gold_numeric = normalize_numeric(gold_normalized)

    if predicted_numeric is not None and gold_numeric is not None:
        return predicted_numeric == gold_numeric

    return False


def evaluate_trajectory(
    trajectory: str,
    gold_answer: str,
) -> dict:
    """
    Extract the predicted answer from a reasoning trajectory
    and evaluate it against the gold answer.
    """

    predicted_answer = extract_boxed_answer(trajectory)

    if predicted_answer is None:
        return {
            "predicted_answer": None,
            "gold_answer": gold_answer,
            "correct": False,
            "valid_output": False,
        }

    correct = answers_match(
        predicted_answer,
        gold_answer,
    )

    return {
        "predicted_answer": predicted_answer,
        "gold_answer": gold_answer,
        "correct": correct,
        "valid_output": True,
    }
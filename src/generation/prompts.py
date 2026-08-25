from textwrap import dedent


SYSTEM_PROMPT = dedent(
    """
    Solve the mathematical problem carefully.

    Provide a clear reasoning process and finish with the final
    answer in the form \\boxed{answer}.
    """
).strip()


def build_prompt(problem: str) -> str:
    """
    Construct the model input for a single mathematical problem.
    """

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Problem:\n{problem}\n\n"
        "Solution:"
    )
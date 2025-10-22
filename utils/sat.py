import random
import argparse
from typing import List
import inspect


ops = ["and", "or", "not", "var", "(", ")"]


def get_valid_tokens(seq: List[str]) -> List[str]:
    p_count = seq.count("(") - seq.count(")")
    if not seq:
        return ["(", "var", "not"]
    if seq[-1] == "var":
        if p_count == 0:
            return ["and", "or"]
        else:
            return ["and", "or", ")"]
    if seq[-1] in ["and", "or"]:
        return ["var", "not", "("]
    if seq[-1] == "not":
        return ["var", "("]
    if seq[-1] == "(":
        return ["var", "not"]
    if seq[-1] == ")":
        return ["and", "or"]


def generate_random_sat_fn(n_vars: int) -> str:
    """Generate a random SAT function including or/and."""
    seq = []
    while n_vars > 0:
        valid_next = get_valid_tokens(seq)
        next_token = random.choice(valid_next)
        if next_token == "var":
            n_vars -= 1
        seq.append(next_token)
    # append a closing parenthesis
    seq.extend([")"] * (seq.count("(") - seq.count(")")))

    # covert to python fn
    i = 0
    for j, token in enumerate(seq):
        if token == "var":
            seq[j] = f"args[{i}]"
            i += 1

    # convert to a function
    ns = {}
    source_code = f"def fn(*args):\n\treturn {' '.join(seq)}"
    exec(source_code, ns)
    return ns["fn"], source_code, seq

def generate_simple_sat_fn(n_vars: int) -> str:
    """Generate a simple SAT function consisting of only required/forbidden features."""
    distance = random.randint(1, n_vars)


def main():
    parser = argparse.ArgumentParser(description="Generate random boolean expressions")
    parser.add_argument(
        "--n-vars",
        "-n",
        type=int,
        default=2,
        help="Number of variables in the expression (default: 2)",
    )

    args = parser.parse_args()
    fn, source_code, seq = generate_random_sat_fn(args.n_vars)
    args = [random.choice([True, False]) for _ in range(args.n_vars)]
    print(source_code)
    print(fn(*args))


if __name__ == "__main__":
    main()

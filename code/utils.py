import os
from data import make_dataset

def output_path(folder, name=""):
    """Create output_<folder>/ next to the scripts and return the path of `name` in it."""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"output_{folder}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)

def save_results(folder, lines):
    """Print the results and save them in output_<folder>/results.txt."""
    text = "\n".join(lines)
    print(text)
    with open(output_path(folder, "results.txt"), "w") as f:
        f.write(text + "\n")
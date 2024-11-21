import json
import sys
from typing import Dict

if __name__ == "__main__":
    lib = sys.argv[1]  # library e.g.  libs/mongodb, libs/langchain-checkpoint-mongodb
    files = sys.argv[2:] # changed files

    dirs_to_run: Dict[str, set] = {
        "lint": set(),
        "test": set(),
    }

    if len(files) == 300:
        # max diff length is 300 files - there are likely files missing
        raise ValueError("Max diff reached. Please manually run CI on changed libs.")

    for file in files:
        if any(
            file.startswith(dir_)
            for dir_ in (
                ".github/workflows",
                ".github/tools",
                ".github/actions",
                ".github/scripts/check_diff.py",
            )
        ):
            # add lib if infra changes
            dirs_to_run["test"].add(lib)
            dirs_to_run["lint"].add(lib)
            # or if any file changed in lib
        if file.startswith(lib):
            dirs_to_run["test"].add(lib)
            dirs_to_run["lint"].add(lib)
    outputs = {
        "dirs-to-lint": list(dirs_to_run["lint"]),
        "dirs-to-test": list(dirs_to_run["test"]),
    }
    for key, value in outputs.items():
        json_output = json.dumps(value)
        print(f"{key}={json_output}")  # noqa: T201

"""The GPU jobs, each run on the box as `python -m dual.jobs.<name>`.

The image bakes no weights: the kyutai repos sit in the shared dir's HF cache. The
shared dir is the job's cwd, so HF_HOME follows it unless the image already set it.
This runs before any job module imports huggingface_hub or moshi.
"""

import os
import pathlib

os.environ.setdefault("HF_HOME", str(pathlib.Path.cwd() / "hf"))

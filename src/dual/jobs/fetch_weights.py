"""Downloads nvidia/personaplex-7b-v1 once into the shared dir. Needs HF_TOKEN (a
--secret)."""

import os
import pathlib

from huggingface_hub import snapshot_download

from dual.constants import PERSONAPLEX_FILES

REPO = "nvidia/personaplex-7b-v1"


def main() -> None:
    os.environ.pop("HF_HUB_OFFLINE", None)
    target = pathlib.Path.cwd() / "models" / "personaplex-7b-v1"
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        REPO,
        local_dir=str(target),
        allow_patterns=[*PERSONAPLEX_FILES.values(), "config.json", "README.md"],
        token=os.environ["HF_TOKEN"],
    )
    for name in PERSONAPLEX_FILES.values():
        path = target / name
        print(name, path.stat().st_size if path.exists() else "MISSING")
    print("fetched", target)


if __name__ == "__main__":
    main()

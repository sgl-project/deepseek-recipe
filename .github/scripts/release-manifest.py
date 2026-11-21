"""Check the release artifacts in <release>/dist and write SHA256SUMS and build-info.json beside them."""

import argparse
import hashlib
import json
import os
import sys
import tomllib
from pathlib import Path

PLATFORMS = (
    "manylinux_2_28_x86_64",
    "manylinux_2_28_aarch64",
    "macosx_11_0_arm64",
    "macosx_11_0_x86_64",
)
UPSTREAM = "https://github.com/deepseek-ai/deepseek-recipe"


def load_toml(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release", type=Path)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--upstream-main-sha", required=True)
    parser.add_argument("--upstream-base-sha", required=True)
    args = parser.parse_args()

    name = load_toml("deepseek-recipe-python/pyproject.toml")["project"]["name"]
    version = load_toml("deepseek-recipe-python/Cargo.toml")["package"]["version"]
    stem = name.replace("-", "_")
    expected = {f"{stem}-{version}.tar.gz"} | {
        f"{stem}-{version}-cp310-abi3-{platform}.whl" for platform in PLATFORMS
    }

    dist = args.release / "dist"
    found = {path.name for path in dist.iterdir()}
    if found != expected:
        sys.exit(
            f"missing {sorted(expected - found)}, unexpected {sorted(found - expected)}"
        )

    files = [
        {
            "name": file,
            "sha256": hashlib.sha256((dist / file).read_bytes()).hexdigest(),
            "size": (dist / file).stat().st_size,
        }
        for file in sorted(found)
    ]
    (args.release / "SHA256SUMS").write_text(
        "".join(f"{f['sha256']}  {f['name']}\n" for f in files)
    )

    server = os.environ["GITHUB_SERVER_URL"]
    repository = os.environ["GITHUB_REPOSITORY"]
    info = {
        "distribution": name,
        "version": version,
        "source_repository": f"{server}/{repository}",
        "source_ref": os.environ["GITHUB_REF"],
        "source_sha": args.source_sha,
        "upstream_repository": UPSTREAM,
        "upstream_main_sha": args.upstream_main_sha,
        "upstream_base_sha": args.upstream_base_sha,
        "rust_toolchain": load_toml("rust-toolchain.toml")["toolchain"]["channel"],
        "workflow_run": f"{server}/{repository}/actions/runs/{os.environ['GITHUB_RUN_ID']}",
        "files": files,
    }
    (args.release / "build-info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

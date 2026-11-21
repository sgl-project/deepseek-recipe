# Releasing sgl-deepseek-recipe

This repository publishes the Python bindings as `sgl-deepseek-recipe`, imported
as `deepseek_recipe`. The Rust crates keep the workspace version; consumers pin
them to a commit of this repository.

## Workflows

- [CI](../.github/workflows/ci.yml) runs on pull requests into `dev` and on
  pushes to `dev`. It builds the minimal OpenCV 4.12.0 of
  [Building the Python wheel](build-python-wheel.md), caches it, and runs the
  [Rust checks](development.md#rust-checks) and the
  [Python tests](development.md#python-development-and-tests) for the whole
  workspace.
- [Wheels](../.github/workflows/wheels.yml) runs on manual dispatch, and on pull
  requests into `dev` that change the wheel build. It builds:
  - the sdist, and a wheel from that sdist to prove that it builds;
  - `cp310-abi3` wheels for `manylinux_2_28_x86_64` and `manylinux_2_28_aarch64`
    inside the manylinux images, and for `macosx_11_0_arm64` and
    `macosx_11_0_x86_64`, each repaired with `auditwheel` or `delocate`.

  The minimal build disables AVIF so macOS wheels do not inherit unrelated
  multimedia libraries with newer deployment targets.

  Each wheel is installed by name in a fresh environment on Python 3.10 and
  3.14, on a runner without OpenCV, and runs the binding tests. The `release`
  artifact holds the distributions, `SHA256SUMS`, and `build-info.json`, which
  records the source commit, the upstream `deepseek-ai/deepseek-recipe` `main`
  commit and merge base, the Rust toolchain, and the workflow run.

  With `publish` set, a dispatch on `dev` or an `sgl-v*` tag uploads that
  artifact to TestPyPI and then, once the `pypi` environment approves, to PyPI.
  Both uploads use trusted publishing and verify `SHA256SUMS` first.

## One-time setup

Done by a repository admin and a PyPI owner, in this order:

1. Make `dev` the default branch. GitHub offers a manual dispatch only for a
   workflow on the default branch.

   ```sh
   gh api -X PATCH repos/sgl-project/deepseek-recipe -f default_branch=dev
   ```

2. Create the `testpypi` and `pypi` environments under **Settings →
   Environments**. Limit both to the `dev` branch and `sgl-v*` tags, and require
   a release reviewer on `pypi`. Create them before the trusted publishers: a
   run that names a missing environment creates it without protection.
3. On PyPI, under **Your projects → Publishing → Add a new pending publisher**,
   register project `sgl-deepseek-recipe`, owner `sgl-project`, repository
   `deepseek-recipe`, workflow `wheels.yml`, environment `pypi`. Register the
   same on TestPyPI with environment `testpypi`. The first upload creates the
   project; then add the other maintainers as owners.

The workflows need no secrets. Every job but the two uploads runs with a
read-only token.

## Release

1. On a branch from `dev`, set `version` in
   [deepseek-recipe-python/Cargo.toml](../deepseek-recipe-python/Cargo.toml), run
   `cargo update --workspace`, and move the `Unreleased` entries of the
   [changelog](../CHANGELOG.md) under the new version. Merge it into `dev`.
2. Dispatch the build and upload:

   ```sh
   gh workflow run wheels.yml --repo sgl-project/deepseek-recipe --ref dev -f publish=true
   ```

3. When the `testpypi` job finishes, install the release from TestPyPI on Linux
   and macOS and run the binding tests, then approve the `pypi` deployment:

   ```sh
   python3 -m venv /tmp/rc
   /tmp/rc/bin/pip install --no-deps --index-url https://test.pypi.org/simple/ \
       'sgl-deepseek-recipe==<version>'
   /tmp/rc/bin/pip install 'pytest>=8,<10'
   /tmp/rc/bin/python -m pytest deepseek-recipe-python/tests -q
   ```

4. Tag the source commit of `build-info.json` and attach the manifest:

   ```sh
   gh run download <run-id> --repo sgl-project/deepseek-recipe -n release -D release
   sha=$(jq -r .source_sha release/build-info.json)
   gh release create sgl-v<version> --repo sgl-project/deepseek-recipe --target "$sha" \
       --title "sgl-deepseek-recipe <version>" release/build-info.json release/SHA256SUMS
   ```

A version that reaches PyPI cannot be uploaded again; a failed or rejected
release takes the next version.

# Releasing sgl-deepseek-recipe

This repository publishes the Python bindings to PyPI as `sgl-deepseek-recipe`,
imported as `deepseek_recipe`, and four Rust crates to crates.io at the
workspace version; see [Rust crates](#rust-crates). The two have separate
version series and release procedures.

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
- [Rust release](../.github/workflows/rust-release.yml) runs on manual dispatch
  only, in the `release` environment. It packages the four
  [Rust crates](#rust-crates) together, then publishes them one at a time in
  dependency order. `dry_run`, set by default, stops before the upload and runs
  without a credential; `auth` selects the [credential](#credentials) of an
  upload.

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

CI and Wheels need no secrets, and every job of theirs but the two uploads runs
with a read-only token. Rust release reads one secret, for the first upload of
a crate; see [Credentials](#credentials).

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

## Rust crates

Four packages are published to crates.io. Each keeps its library name, so
`use deepseek_recipe::…` is unchanged for consumers.

| Package | Library | Directory |
| --- | --- | --- |
| `sgl-deepseek-recipe-macros` | `deepseek_recipe_macros` | `deepseek-recipe-macros` |
| `sgl-deepseek-recipe-core` | `deepseek_recipe_core` | `deepseek-recipe-core` |
| `sgl-deepseek-recipe-encoding` | `deepseek_recipe_encoding` | `deepseek-recipe-encoding` |
| `sgl-deepseek-recipe` | `deepseek_recipe` | `deepseek-recipe` |

Always select these four with `-p`. `deepseek-recipe-image` and
`deepseek-recipe-python` are workspace members that are not published to
crates.io, and `cargo publish` without `-p`, or with `--workspace`, selects them
too.

### Credentials

- Create the `release` environment under the repository's **Settings →
  Environments** before the first dispatch, limited like `pypi` above.
- The first upload of each package needs a crates.io API token of an SGLang
  owner, because Trusted Publishing can only be configured for a crate that
  already exists. Store it as the repository secret
  `SGL_DEEPSEEK_RECIPE_CRATES_IO_TOKEN` and dispatch with `auth=token`; do not
  write it to a file in the repository or to a workflow. After the upload, add
  the other maintainers as owners, revoke the token, and delete the secret.
- Later releases dispatch with `auth=oidc`, which uses Trusted Publishing. On
  crates.io, under each crate's **Settings → Trusted Publishing**, add a GitHub
  publisher with repository owner `sgl-project`, repository name
  `deepseek-recipe`, workflow filename `rust-release.yml`, and environment
  `release`.

### Release

1. On a branch from `dev`, set `version` under `[workspace.package]` and the
   `version` of the four `sgl-*` entries under `[workspace.dependencies]` in
   [Cargo.toml](../Cargo.toml), run `cargo update --workspace`, and move the
   `Unreleased` entries of the [changelog](../CHANGELOG.md) under the new
   version. Merge it into `dev`.
2. Dispatch a dry run on the merged commit:

   ```sh
   gh workflow run rust-release.yml --repo sgl-project/deepseek-recipe --ref dev
   ```

   With the pinned toolchain it packages the four together, then runs
   `cargo publish --dry-run` with the same selection:

   ```sh
   cargo package --locked \
       -p sgl-deepseek-recipe-macros -p sgl-deepseek-recipe-core \
       -p sgl-deepseek-recipe-encoding -p sgl-deepseek-recipe
   ```

   Cargo builds each packaged crate against the others. Selected alone, a crate
   whose dependency is not on crates.io at the new version fails with
   `no matching package named …`, so a dry run cannot go one crate at a time.
3. Publish, with `auth=token` instead for the first upload of a crate:

   ```sh
   gh workflow run rust-release.yml --repo sgl-project/deepseek-recipe --ref dev \
       -f dry_run=false -f auth=oidc
   ```

   After packaging the four together, it runs one command per crate, in this
   order, and stops at the first failure:

   ```sh
   cargo publish --locked -p sgl-deepseek-recipe-macros
   cargo publish --locked -p sgl-deepseek-recipe-core
   cargo publish --locked -p sgl-deepseek-recipe-encoding
   cargo publish --locked -p sgl-deepseek-recipe
   ```

   Each command resolves the crate's `sgl-*` dependencies on crates.io, builds
   it, uploads it, and waits until the new version is in the index; only then
   does the next command start. Crates uploaded before a failure stay
   published, and a second dispatch cannot get past them because their version
   already exists. If a command times out while waiting, check the crate's
   page on crates.io. Then publish the remaining crates with the commands
   above, in order, from a clean checkout of the same commit with
   `CARGO_REGISTRY_TOKEN` set to an owner's token.
4. Verify from the registry alone, in a new directory outside this repository,
   with no `path` dependencies and no `[patch]`:

   ```sh
   cargo new --edition 2024 recipe-registry-check
   cd recipe-registry-check
   cargo add sgl-deepseek-recipe@<version> --rename deepseek-recipe
   cargo add sgl-deepseek-recipe-core@<version> --rename deepseek-recipe-core
   cargo add sgl-deepseek-recipe-encoding@<version> sgl-deepseek-recipe-macros@<version> serde_json@1
   ```

   Save the Rust example of the [README](../README.md#rust) as `src/main.rs`,
   add a `deepseek_recipe_core::prompt!("hi")` call to it, and run `cargo run`.
   In the generated `Cargo.lock`, the four `sgl-*` packages must have a
   `registry+` source at the new version, and no package may be named
   `deepseek-recipe…`. Then check that the four docs.rs pages have built.

A version that reaches crates.io cannot be replaced, only yanked; a failed or
wrong release takes the next version.

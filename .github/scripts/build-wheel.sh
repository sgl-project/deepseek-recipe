#!/usr/bin/env bash
# Follows docs/build-python-wheel.md. Runs on macOS, or on Linux inside
# quay.io/pypa/manylinux_2_28_<arch>, whose dnf and /opt/python it uses.
set -euxo pipefail

root=$PWD
opencv=$root/target/opencv-min

case "$(uname -s)" in
Linux)
    # The bindings of opencv 0.93 do not compile when libclang 22 generates them;
    # 18 is also the clang of the CI runner.
    dnf install -y clang-18.1.8 clang-devel-18.1.8
    python=/opt/python/cp312-cp312/bin/python3
    repair=auditwheel
    ;;
Darwin)
    export MACOSX_DEPLOYMENT_TARGET=11.0
    python=python3
    repair=delocate
    ;;
esac

if ! command -v rustup >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
        sh -s -- -y --profile minimal --default-toolchain none
    . "$HOME/.cargo/env"
fi
rustup toolchain install
rustc --version

"$python" -m venv "$root/target/wheel-venv"
. "$root/target/wheel-venv/bin/activate"
pip install 'maturin>=1.9,<2' 'twine>=7' "$repair"
pip list

.github/scripts/build-opencv.sh "$opencv"
export OPENCV_INCLUDE_PATHS=$opencv/include/opencv4
export OPENCV_LINK_PATHS=$opencv/lib
export OPENCV_LINK_LIBS=opencv_core,opencv_imgproc,opencv_imgcodecs
export RUSTFLAGS="-L$opencv/lib -C link-arg=-Wl,-rpath,$opencv/lib"
unset CARGO_ENCODED_RUSTFLAGS

rm -rf "$root/target/wheels" "$root/wheelhouse"
maturin build --release --locked \
    --manifest-path deepseek-recipe-python/Cargo.toml --out "$root/target/wheels"

case "$(uname -s)" in
Linux)
    auditwheel repair --plat "manylinux_2_28_$(uname -m)" \
        -w "$root/wheelhouse" "$root"/target/wheels/*.whl
    auditwheel show "$root"/wheelhouse/*.whl
    ;;
Darwin)
    delocate-wheel --require-archs "$(uname -m)" \
        -w "$root/wheelhouse" "$root"/target/wheels/*.whl
    delocate-listdeps --all "$root"/wheelhouse/*.whl
    ;;
esac
twine check --strict "$root"/wheelhouse/*.whl

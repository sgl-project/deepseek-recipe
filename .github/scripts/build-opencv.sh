#!/usr/bin/env bash
# The minimal OpenCV of docs/build-python-wheel.md; keep the options in step with it.
# CMAKE_INSTALL_LIBDIR keeps the libraries in <prefix>/lib on the RHEL-based manylinux images too.
set -euxo pipefail

prefix=$1
src=$(mktemp -d)

git clone --depth 1 --branch 4.12.0 https://github.com/opencv/opencv.git "$src"
cmake -S "$src" -B "$src/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_LIST=core,imgproc,imgcodecs \
    -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DBUILD_ZLIB=ON -DBUILD_JPEG=ON -DBUILD_PNG=ON -DBUILD_WEBP=ON \
    -DWITH_JPEG=ON -DWITH_PNG=ON -DWITH_WEBP=ON \
    -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_apps=OFF -DBUILD_DOCS=OFF -DBUILD_JAVA=OFF \
    -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF \
    -DBUILD_PROTOBUF=OFF \
    -DWITH_ADE=OFF -DWITH_AVIF=OFF -DWITH_EIGEN=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF \
    -DWITH_GTK=OFF -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_JASPER=OFF \
    -DWITH_LAPACK=OFF -DWITH_OPENCL=OFF -DWITH_OPENCLAMDBLAS=OFF \
    -DWITH_OPENCLAMDFFT=OFF -DWITH_OPENEXR=OFF -DWITH_OPENJPEG=OFF \
    -DWITH_OPENMP=OFF -DWITH_QUIRC=OFF -DWITH_TBB=OFF -DWITH_TIFF=OFF \
    -DWITH_V4L=OFF -DWITH_1394=OFF
cmake --build "$src/build" --parallel "$(getconf _NPROCESSORS_ONLN)"
cmake --install "$src/build"
rm -rf "$src"

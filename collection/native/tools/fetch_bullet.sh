#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
test -d vendor/bullet3-3.25 && exit 0
mkdir -p vendor
archive=vendor/bullet3-3.25.tar.gz
curl --fail --location https://github.com/bulletphysics/bullet3/archive/refs/tags/3.25.tar.gz --output "$archive"
printf '%s  %s\n' c45afb6399e3f68036ddb641c6bf6f552bf332d5ab6be62f7e6c54eda05ceb77 "$archive" | shasum -a 256 -c -
tar -xzf "$archive" -C vendor

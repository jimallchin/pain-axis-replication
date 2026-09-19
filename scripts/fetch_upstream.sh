#!/bin/bash
# Pinned checkout of the released Pain-axis repository into external/Pain-axis.
# The trial logs, scenario bank and pain vectors are read from there.
set -e
COMMIT=8d1649c03a63a39c9aa092532c376800cc4a3863
cd "$(dirname "$0")/.."
mkdir -p external
if [ ! -d external/Pain-axis/.git ]; then
  git clone --quiet https://github.com/valen-research/Pain-axis external/Pain-axis
fi
git -C external/Pain-axis fetch --quiet origin
echo "upstream HEAD: $(git -C external/Pain-axis rev-parse origin/HEAD)"
git -C external/Pain-axis checkout --quiet "$COMMIT"
echo "checked out:   $(git -C external/Pain-axis rev-parse HEAD)"

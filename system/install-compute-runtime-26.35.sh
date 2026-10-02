#!/bin/bash
# Installs Intel compute-runtime 26.35.39758.10 + IGC 2.41.5 from the official GitHub releases,
# sha256-verified. 26.35 is required: it fixes the multi-device host-RAM mirror (intel/compute-runtime#980)
# that makes TP=2 OOM a 30 GiB host, and vllm-xpu-kernels >= 0.1.10 needs >= 26.18 anyway.
# Ubuntu 26.04 ships 26.05; ppa:kobuk-team/intel-graphics has 26.31 (install it first for xpu-smi etc.).
set -e
D=$(mktemp -d); cd "$D"
B=https://github.com/intel/compute-runtime/releases/download/26.35.39758.10
G=https://github.com/intel/intel-graphics-compiler/releases/download/v2.41.5
wget -q $G/intel-igc-core-2_2.41.5+22716_amd64.deb $G/intel-igc-opencl-2_2.41.5+22716_amd64.deb \
  $B/intel-ocloc_26.35.39758.10-0_amd64.deb $B/intel-opencl-icd_26.35.39758.10-0_amd64.deb \
  $B/libze-intel-gpu1_26.35.39758.10-0_amd64.deb $B/ww35.sum
grep -v dbgsym ww35.sum | grep -E "ocloc_|opencl-icd_|libze-intel-gpu1_" | sha256sum -c
cat > igc.sum <<'SUM'
0a6e64a663ae65a0fa02d6912ae3b6b37cf85b90c21cc423fd9fef70aaf4f628  intel-igc-core-2_2.41.5+22716_amd64.deb
779e1b9e88098eb25711e9a8f67c2752665bad22f134aa40ed5649f6e1b87058  intel-igc-opencl-2_2.41.5+22716_amd64.deb
SUM
sha256sum -c igc.sum
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ./intel-igc-core-2_*.deb ./intel-igc-opencl-2_*.deb \
  ./intel-ocloc_*.deb ./intel-opencl-icd_*.deb ./libze-intel-gpu1_*.deb
clinfo -l

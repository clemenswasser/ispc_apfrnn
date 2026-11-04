#!/usr/bin/env bash
# Isolated ISPC kernel inspection harness.
#
# Usage:
#   scripts/ispc_analyze.sh [output-directory] [kernel-name ...]
#
# With no kernel names, all neighbor-list writers are analyzed. For each kernel
# the script writes:
#   <name>.s        extracted assembly function
#   <name>.mca.txt  llvm-mca throughput / bottleneck report
#   <name>.mca.err  llvm-mca warnings
#
# Environment:
#   ISPC_TARGET  ISPC target (default: avx2vnni-i32x8)
#   MCPU         llvm-mca CPU model (default: alderlake)
#   MCA_ITERS    llvm-mca iteration count (default: 100)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-${root}/build/ispc-analysis}"
if [ "$#" -gt 0 ]; then
  shift
fi

target="${ISPC_TARGET:-avx2vnni-i32x8}"
cpu="${MCPU:-alderlake}"
iterations="${MCA_ITERS:-100}"

for tool in ispc llvm-mca llvm-objdump; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "error: ${tool} is required" >&2
    exit 1
  fi
done

mkdir -p "${out}"
# The same flags as CMakeLists.txt.
common=(--target="${target}" -O3 --opt=fast-math --addressing=64
        --enable-llvm-intrinsics)

# One assembly listing over all components (src/kernels/*.ispc).
: > "${out}/kernel.s"
: > "${out}/kernel.disasm.s"
for src in "${root}"/src/kernels/*.ispc; do
  name="$(basename "${src}" .ispc)"
  echo "==> ${src} (target ${target})"
  ispc "${src}" "${common[@]}" --emit-asm -o "${out}/${name}.s"
  ispc "${src}" "${common[@]}" --emit-llvm-text -o "${out}/${name}.ll"
  ispc "${src}" "${common[@]}" --emit-obj -o "${out}/${name}.o"
  cat "${out}/${name}.s" >> "${out}/kernel.s"
  llvm-objdump -d --no-show-raw-insn "${out}/${name}.o" \
    >> "${out}/kernel.disasm.s"
done

kernels=("$@")
if [ "${#kernels[@]}" -eq 0 ]; then
  kernels=(
    write_fixed_rows
    write_fixed_rows_oct
    write_fixed_rows_dense
    hash_stencil_ranges
    csr_rows_pair
  )
fi

for kernel in "${kernels[@]}"; do
  block="${out}/${kernel}.s"
  if ! grep -q -- "-- Begin function ${kernel}___" "${out}/kernel.s"; then
    echo "warning: ${kernel} not found in ${out}/kernel.s" >&2
    continue
  fi

  awk -v pattern="${kernel}___" '
    /-- Begin function / { capture = index($0, pattern) != 0 }
    capture { print }
    /-- End function/ { if (capture) exit }
  ' "${out}/kernel.s" > "${block}"

  echo "==> llvm-mca ${kernel}"
  if llvm-mca -mcpu="${cpu}" -iterations="${iterations}" \
      -bottleneck-analysis "${block}" \
      > "${out}/${kernel}.mca.txt" 2> "${out}/${kernel}.mca.err"; then
    sed -n '1,12p' "${out}/${kernel}.mca.txt"
  else
    echo "warning: llvm-mca failed for ${kernel} (see ${kernel}.mca.err)" >&2
  fi
done

echo
echo "artifacts in ${out}:"
ls -1 "${out}" | sed 's/^/  /'

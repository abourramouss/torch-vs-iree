#!/bin/bash
# Compare per-dispatch MLIR between 02_iree (O0) and 03_iree_optimized (O3).
#
# Usage:
#   ./diff_mlirs.sh              # list which dispatches differ
#   ./diff_mlirs.sh summary      # same, with a count
#   ./diff_mlirs.sh full         # unified diff of all differing dispatches
#   ./diff_mlirs.sh <name>       # unified diff for a specific dispatch
#                                # e.g. ./diff_mlirs.sh dispatch_12
#
# Requires both 02_iree/compile.sh and 03_iree_optimized/compile.sh to have
# been run (they populate 02_iree/dumps/ and 03_iree_optimized/dumps/).

set -e
cd "$(dirname "$0")"

A=02_iree/dumps
B=03_iree_optimized/dumps

if [ ! -d "$A" ] || [ ! -d "$B" ]; then
  echo "ERROR: missing dumps. Run 02_iree/compile.sh and 03_iree_optimized/compile.sh first."
  exit 1
fi

mode="${1:-list}"

case "$mode" in
  list|summary)
    diff_count=0
    only_a=0
    only_b=0
    same=0
    # Iterate by dispatch index suffix (ignore filename differences from renamed ops).
    # Key files by the "_dispatch_N" token so renamed ops still line up.
    declare -A by_idx_a by_idx_b
    for f in "$A"/module_main*dispatch_*.mlir; do
      idx=$(basename "$f" | sed -E 's/.*_dispatch_([0-9]+).*/\1/')
      by_idx_a[$idx]="$f"
    done
    for f in "$B"/module_main*dispatch_*.mlir; do
      idx=$(basename "$f" | sed -E 's/.*_dispatch_([0-9]+).*/\1/')
      by_idx_b[$idx]="$f"
    done
    all_idx=$(printf "%s\n%s\n" "${!by_idx_a[@]}" "${!by_idx_b[@]}" | sort -un)
    for idx in $all_idx; do
      fa="${by_idx_a[$idx]:-}"
      fb="${by_idx_b[$idx]:-}"
      if [ -z "$fa" ]; then
        echo "+ dispatch_$idx   (only in O3: $(basename "$fb"))"
        only_b=$((only_b+1))
      elif [ -z "$fb" ]; then
        echo "- dispatch_$idx   (only in O0: $(basename "$fa"))"
        only_a=$((only_a+1))
      elif ! diff -q "$fa" "$fb" >/dev/null 2>&1; then
        echo "~ dispatch_$idx   $(basename "$fa")  <>  $(basename "$fb")"
        diff_count=$((diff_count+1))
      else
        same=$((same+1))
      fi
    done
    if [ "$mode" = "summary" ]; then
      total=$(echo "$all_idx" | wc -w)
      echo
      echo "Summary: $total total dispatches"
      echo "  identical: $same"
      echo "  differing: $diff_count"
      echo "  only in O0: $only_a"
      echo "  only in O3: $only_b"
    fi
    ;;
  full)
    diff -ru "$A" "$B" | less -R
    ;;
  *)
    # Treat as a dispatch name or index
    target="$1"
    fa=$(ls "$A"/*${target}*.mlir 2>/dev/null | head -1 || true)
    fb=$(ls "$B"/*${target}*.mlir 2>/dev/null | head -1 || true)
    if [ -z "$fa" ] || [ -z "$fb" ]; then
      echo "ERROR: could not find dispatch matching '$target' in both dumps" >&2
      exit 1
    fi
    echo "--- $fa"
    echo "+++ $fb"
    diff -u "$fa" "$fb" || true
    ;;
esac

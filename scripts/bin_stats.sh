#!/usr/bin/env bash
# Summarize masked-peak m/z-bin predictions from a training stdout log.
# Usage: scripts/bin_stats.sh <stdout_file> [top_n]
set -euo pipefail

FILE="${1:?usage: $0 <stdout_file> [top_n]}"
TOP="${2:-15}"
[[ -f "$FILE" ]] || { echo "No such file: $FILE" >&2; exit 1; }

tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT

# Rows: "<idx> <true_bin> <I_true> <pred_bin> ...". tqdm text can concatenate onto
# the last column, so take the leading numeric token of each field.
awk '
  function num(s){ return (match(s, /^-?[0-9]+\.?[0-9]*/) ? substr(s, RSTART, RLENGTH) : "") }
  $1 ~ /^[0-9]+$/ && NF >= 4 {
    t=num($2); p=num($4)
    if (t=="" || p=="") next
    print t, p
  }
' "$FILE" > "$tmp"

total=$(wc -l < "$tmp" | tr -d ' ')
[[ "$total" -gt 0 ]] || { echo "No prediction-table rows found in $FILE" >&2; exit 1; }

echo "File: $FILE"
echo "Parsed masked-peak rows: $total"
echo "Distinct predicted bins: $(cut -d' ' -f2 "$tmp" | sort -u | wc -l | tr -d ' ')  |  Distinct true bins: $(cut -d' ' -f1 "$tmp" | sort -u | wc -l | tr -d ' ')"
echo

echo "== Top $TOP PREDICTED bins =="
printf "%-10s %10s %8s\n" bin count pct
cut -d' ' -f2 "$tmp" | sort | uniq -c | sort -rn | head -n "$TOP" \
  | awk -v tot="$total" '{printf "%-10s %10d %7.2f%%\n", $2, $1, 100*$1/tot}'
echo

echo "== Top $TOP TRUE bins (-1 = ignored padding) =="
printf "%-10s %10s %8s\n" bin count pct
cut -d' ' -f1 "$tmp" | sort | uniq -c | sort -rn | head -n "$TOP" \
  | awk -v tot="$total" '{printf "%-10s %10d %7.2f%%\n", $2, $1, 100*$1/tot}'
echo

echo "== val_acc_mz_bin =="
{ grep -oE 'val_acc_mz_bin=[0-9.]+' "$FILE" || true; } | sed 's/.*=//' | awk '
  NR==1 { first=$1; max=$1 }
  { last=$1; if ($1+0 > max+0) max=$1; n++ }
  END {
    if (n) printf "count: %d   first: %s   last: %s   max: %s\n", n, first, last, max
    else print "(none found)"
  }'

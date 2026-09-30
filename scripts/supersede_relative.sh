#!/usr/bin/env bash
# One-off (2026-09-30): AQI and Synthetic were first trained level-relative; ADF shows they
# are stationary, so --param auto trains them in levels. Keep the old runs out of the
# tables: asgtm becomes the 'asgtm-relative' ablation, the rest moves to _superseded_relative/.
set -euo pipefail
cd "${1:-results}"
for ds in AirQuality Synthetic; do
  [[ -d $ds ]] || continue
  mkdir -p "_superseded_relative/$ds"
  [[ -d $ds/asgtm && ! -d $ds/asgtm-relative ]] && mv "$ds/asgtm" "$ds/asgtm-relative"
  for d in "$ds"/*/; do
    name=$(basename "$d")
    case $name in
      persistence|linear|var|locf|interp|asgtm-relative) ;;
      *) mv "$d" "_superseded_relative/$ds/" ;;
    esac
  done
done
[[ -d Exchange/asgtm-level ]] && mkdir -p _superseded_relative/Exchange && mv Exchange/asgtm-level _superseded_relative/Exchange/
exit 0

#!/usr/bin/env bash
set -euo pipefail

# Optimise components to fit measured data, then model (no S3 required).
# Usage: ./scripts/run_optimise.sh [csv_file]
#   csv_file: two columns per line, no header — temperature (K), heat capacity
#
# The component arrays below are the initial guesses the optimiser starts from.
# Assets are written to ./data/output/; fitted parameters are printed as value
# outputs in the Output JSON.

CSV_FILE="${1:-data/input.csv}"

DEBYE='[{"component":90,"prefactor":0.8}]'
EINSTEIN='[{"component":250,"prefactor":0.2}]'
LINEAR=0
START_TEMP=2
END_TEMP=50

uv run model-run --local \
  --parameters "$(jq -n \
    --arg file "file://${CSV_FILE}" \
    --argjson debye "${DEBYE}" \
    --argjson einstein "${EINSTEIN}" \
    --argjson linear "${LINEAR}" \
    --argjson start "${START_TEMP}" \
    --argjson end "${END_TEMP}" \
    '{mode:"optimise",
      inputs:{
        data:{uri:$file, mime_type:"text/csv"},
        debye_components:{value:$debye},
        einstein_components:{value:$einstein},
        linear_component:{value:$linear},
        start_temp:{value:$start},
        end_temp:{value:$end}
      },
      parameters:{exponent:1, log_x:false, log_y:false}}')"

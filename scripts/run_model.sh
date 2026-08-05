#!/usr/bin/env bash
set -euo pipefail

# Model heat capacity from fixed components (no S3 required).
# Usage: ./scripts/run_model.sh [csv_file]
#   csv_file: two columns per line, no header — temperature (K), heat capacity
#
# Edit the component arrays below to change the model. Assets are written to
# ./data/output/ (plot.png, model.csv).

CSV_FILE="${1:-data/input.csv}"

DEBYE='[{"component":100,"prefactor":1}]'
EINSTEIN='[]'
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
    '{mode:"model",
      inputs:{
        data:{uri:$file, mime_type:"text/csv"},
        debye_components:{value:$debye},
        einstein_components:{value:$einstein},
        linear_component:{value:$linear},
        start_temp:{value:$start},
        end_temp:{value:$end}
      },
      parameters:{exponent:1, log_x:false, log_y:false}}')"

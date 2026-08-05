# Heat Capacity

Heat-capacity modelling for the LMDS (Local Model Deployment Service) platform.

Models measured heat-capacity-versus-temperature data as a sum of **Debye**,
**Einstein** and optional **linear (electronic, γ)** contributions, producing a
plot and the modelled curve. It can either evaluate a fixed set of components or
**optimise** them to best fit your data.

This is a conversion of the original Flask app to the LMDS model-template
contract (`schema/model.json` + a single `process` handler). The web/REST layer,
matplotlib display backend, and server-side upload/temp-file handling are gone;
the tool now runs Mode-driven under the shared runner, reading a data file and
writing its plot/CSV as assets.

## Quick Start

```bash
# Model from fixed components (edit the arrays in the script to change them):
scripts/run_model.sh data/input.csv

# Optimise the components to fit the data, starting from initial guesses:
scripts/run_optimise.sh data/input.csv

# Both write data/output/plot.png and data/output/model.csv
```

## Modes

| Mode | Response | Inputs | Outputs |
|------|----------|--------|---------|
| `model` | deferred | data file + components + temp range | `plot.png`, `model.csv`, `warnings` |
| `optimise` | deferred | same (as initial guesses) | `plot.png`, `model.csv`, `fitted_*` components, `warnings` |

**Inputs** (both modes):

| Field | Type | Meaning |
|-------|------|---------|
| `data` | file (CSV) | Two columns per line, **no header**: temperature (K), heat capacity |
| `debye_components` | array of `{component, prefactor}` | Debye temperature Θ_D (K) and fractional weight |
| `einstein_components` | array of `{component, prefactor}` | Einstein temperature Θ_E (K) and fractional weight |
| `linear_component` | number | Linear (γ) coefficient; `0` means no linear term |
| `start_temp`, `end_temp` | number | Temperature range to model over (K) |

**Parameters** (model-level, shared by both modes):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `exponent` | `1` | Model/plot as Cp/T^n (n=0 high-T, 3 low-T fitting) |
| `log_x`, `log_y` | `false` | Log-scale the plot axes |

`optimise` additionally returns `fitted_debye_components`,
`fitted_einstein_components` and `fitted_linear` as value outputs. Prefactors are
constrained to sum to 1 and component temperatures are bounded ≥ 1.75 K.

## Project Structure

```
├── schema/
│   ├── model.json              # This tool's contract (model + optimise modes)
│   └── model.schema.json       # Shared meta-schema (copied from template)
├── src/
│   ├── cmds/                   # Entrypoints — copied from template, unmodified
│   ├── infrastructure/         # Runner/handler/S3/logging — copied, unmodified
│   └── model/
│       ├── __init__.py         # handler = HeatCapacityModel()
│       ├── model.py            # Debye/Einstein/linear physics + fitting + plotting
│       └── types.py            # Auto-generated from model.json
├── data/                       # Sample inputs + run outputs (output/ ignored)
├── scripts/                    # run_model.sh, run_optimise.sh
├── tests/test_model.py
├── Dockerfile
└── pyproject.toml
```

## Local development

```bash
uv sync --extra dev
uv run validate-model schema/model.json
uv run model-types schema/model.json -o src/model/types.py
uv run pytest tests/ -v
```

Direct invocation (what the scripts wrap) — see `data/model.json` and
`data/optimise.json` for full example payloads.

## Docker

```bash
docker build -t heat-capacity:1.0.0 .
docker run --rm -v "$PWD/data:/data" heat-capacity:1.0.0 --local \
  --parameters "$(cat data/model.json)"
```

Runs on the template default **Python 3.12** — this tool is pure
numpy/scipy/matplotlib with no pinned/legacy dependencies.

## A note on the optimiser

The physics and the optimiser's objective function are ported **verbatim** from
the original tool so results match the deployed app. One quirk is preserved: in
the fit objective the modelled curve (already divided by `T^exponent`) is divided
by `T^exponent` a second time before being compared to the data. This is the
original behaviour; changing it would change fitted parameters, so it was left
intact. Flag it upstream if the fit should instead minimise a straight residual
on Cp/T^n.

## Environment Variables

Copy `.env.example` to `.env` and adjust for remote (S3) runs:

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `DEBUG` | Logging verbosity |
| `S3_ENDPOINT` | `http://minio:9000` | S3-compatible storage endpoint |
| `S3_ACCESS_KEY` | `minioadmin` | S3 access key |
| `S3_SECRET_KEY` | `minioadmin` | S3 secret key |
| `S3_BUCKET` | `lmds` | S3 bucket name |

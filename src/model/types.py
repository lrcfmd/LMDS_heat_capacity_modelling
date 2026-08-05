# Auto-generated from model.json - DO NOT EDIT
# Regenerate with: uv run model-types <path-to-model.json> -o src/model/types.py

from pathlib import Path
from typing import TypedDict, Literal

class Parameters(TypedDict, total=False):
    # Data is modelled and plotted as Cp/T^n (n=0 for high-T fitting, 3 for low-T fitting). This is n.
    exponent: int  # default: 1
    # Use a logarithmic x (temperature) axis on the plot
    log_x: bool  # default: False
    # Use a logarithmic y (heat capacity) axis on the plot
    log_y: bool  # default: False

class ModelInputs(TypedDict, total=False):
    # Measured data, two columns per line with no header: temperature (K), heat capacity
    data: tuple[Path, str]
    # Debye components, each a Debye temperature and a prefactor (fractional weight)
    debye_components: list[dict]  # default: []
    # Einstein components, each an Einstein temperature and a prefactor (fractional weight)
    einstein_components: list[dict]  # default: []
    # Linear (electronic, γ) coefficient. 0 or omitted means no linear term.
    linear_component: float  # default: 0
    # Temperature to start modelling from, in Kelvin (not well defined below ~1.6 K)
    start_temp: float
    # Temperature to finish modelling at, in Kelvin
    end_temp: float

class OptimiseInputs(TypedDict, total=False):
    # Measured data, two columns per line with no header: temperature (K), heat capacity
    data: tuple[Path, str]
    # Initial-guess Debye components, each a Debye temperature and a prefactor
    debye_components: list[dict]  # default: []
    # Initial-guess Einstein components, each an Einstein temperature and a prefactor
    einstein_components: list[dict]  # default: []
    # Initial-guess linear (electronic, γ) coefficient. 0 or omitted means no linear term.
    linear_component: float  # default: 0
    # Temperature to start modelling from, in Kelvin (not well defined below ~1.6 K)
    start_temp: float
    # Temperature to finish modelling at, in Kelvin
    end_temp: float
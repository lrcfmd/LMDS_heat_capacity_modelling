import csv
from pathlib import Path

import numpy as np
import pytest

from infrastructure.logging import newLogger
from model.model import (
    Debye,
    Einstein,
    HeatCapacityModel,
    create_param_list,
    param_list_seperate,
    parameter_constrainer,
)

LOG = newLogger("test")


@pytest.fixture(scope="module")
def model() -> HeatCapacityModel:
    return HeatCapacityModel()


@pytest.fixture
def data_file(tmp_path: Path) -> Path:
    """A pure Debye (Td=100) + Einstein (Te=300) curve, headerless T,Cp."""
    path = tmp_path / "cp.csv"
    with path.open("w") as f:
        for T in np.arange(2.0, 50.0, 1.5):
            cp = 0.85 * Debye(T, 100.0) + 0.15 * Einstein(T, 300.0)
            f.write(f"{round(float(T), 2)},{round(float(cp), 5)}\n")
    return path


def _inputs(**over):
    base = {
        "debye_components": [{"component": 100, "prefactor": 0.85}],
        "einstein_components": [{"component": 300, "prefactor": 0.15}],
        "linear_component": 0,
        "start_temp": 2,
        "end_temp": 50,
    }
    base.update(over)
    return base


# --- physics unit tests ----------------------------------------------------

def test_debye_positive_and_increasing():
    # Debye Cp rises with temperature below the Debye temperature.
    assert Debye(10.0, 100.0) > 0
    assert Debye(40.0, 100.0) > Debye(10.0, 100.0)


def test_param_roundtrip_with_linear():
    debye = [(100.0, 0.5)]
    einstein = [(300.0, 0.3)]
    linear = 0.2
    flat = create_param_list(debye, einstein, linear)
    d, e, lin = param_list_seperate(flat, n_debye=1)
    assert d == debye
    assert e == einstein
    assert lin == linear


def test_parameter_constrainer_sums_to_one():
    debye = [(100.0, 2.0)]
    einstein = [(300.0, 2.0)]
    d, e, lin = parameter_constrainer(debye, einstein, None)
    total = sum(x[1] for x in d + e)
    assert total == pytest.approx(1.0)


# --- model mode ------------------------------------------------------------

def test_model_writes_assets(model, data_file, tmp_path):
    out = tmp_path / "out"
    result = model.process(
        "model", _inputs(), {"data": (data_file, "text/csv")}, out, {}, LOG
    )
    assert (out / "plot.png").exists()
    assert (out / "model.csv").exists()
    assert "warnings" in result
    # CSV has T, Total and at least the Debye/Einstein columns.
    with (out / "model.csv").open() as f:
        header = next(csv.reader(f))
    assert header[0] == "T"
    assert "Total" in header


def test_model_low_temp_warning(model, data_file, tmp_path):
    result = model.process(
        "model", _inputs(start_temp=1.0),
        {"data": (data_file, "text/csv")}, tmp_path / "o", {}, LOG,
    )
    assert any("1.6 Kelvin" in w for w in result["warnings"])


def test_model_prefactor_warning(model, data_file, tmp_path):
    result = model.process(
        "model",
        _inputs(debye_components=[{"component": 100, "prefactor": 0.5}],
                einstein_components=[]),
        {"data": (data_file, "text/csv")}, tmp_path / "o", {}, LOG,
    )
    assert any("sum to 1" in w for w in result["warnings"])


def test_missing_data_file(model, tmp_path):
    with pytest.raises(ValueError, match="No input data file"):
        model.process("model", _inputs(), {}, tmp_path / "o", {}, LOG)


def test_unknown_mode(model, data_file, tmp_path):
    with pytest.raises(ValueError, match="Unknown mode"):
        model.process(
            "bogus", _inputs(), {"data": (data_file, "text/csv")},
            tmp_path / "o", {}, LOG,
        )


# --- optimise mode ---------------------------------------------------------

def test_optimise_returns_fitted_params(model, data_file, tmp_path):
    out = tmp_path / "out"
    result = model.process(
        "optimise",
        _inputs(debye_components=[{"component": 90, "prefactor": 0.8}],
                einstein_components=[{"component": 250, "prefactor": 0.2}]),
        {"data": (data_file, "text/csv")}, out, {}, LOG,
    )
    assert (out / "plot.png").exists()
    assert (out / "model.csv").exists()
    assert len(result["fitted_debye_components"]) == 1
    assert len(result["fitted_einstein_components"]) == 1
    # Fitted prefactors are constrained to sum to 1.
    total = (
        sum(c["prefactor"] for c in result["fitted_debye_components"])
        + sum(c["prefactor"] for c in result["fitted_einstein_components"])
    )
    assert total == pytest.approx(1.0, abs=1e-2)
    # Each fitted component temperature respects the optimiser's lower bound.
    for c in result["fitted_debye_components"] + result["fitted_einstein_components"]:
        assert c["component"] >= 1.75

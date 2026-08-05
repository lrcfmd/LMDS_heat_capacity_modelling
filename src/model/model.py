import csv
import math
from math import e, exp
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend — no display, render straight to file
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.constants import R  # noqa: E402
from scipy.integrate import quad  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from infrastructure.logging import newLogger  # noqa: E402


# --- Physics ---------------------------------------------------------------
# Ported verbatim from the original LMDS Flask app so results are unchanged.

def Einstein(T, Te):
    """Einstein heat-capacity contribution at temperature T for Einstein temp Te."""
    return (3 * R * (np.power((Te / T), 2))) * (
        (np.power(e, (Te / T))) / (np.power((np.power(e, (Te / T)) - 1), 2))
    )


def _debye_integrand(x):
    return ((np.power(x, 4)) * (exp(x))) / (np.power(exp(x) - 1, 2))


def Debye(T, Td):
    """Debye heat-capacity contribution at temperature T for Debye temp Td."""
    return (9 * R * (np.power(T / Td, 3))) * quad(_debye_integrand, 0, Td / T)[0]


def y(gamma, T):
    """Linear (electronic) contribution."""
    return gamma * T


def param_list_seperate(params, n_debye):
    """Split a flat parameter vector into Debye pairs, Einstein pairs and linear."""
    debye_comps = [(params[i], params[i + 1]) for i in range(0, n_debye * 2, 2)]
    assert len(debye_comps) == n_debye
    einstein_comps = []
    if len(params) % 2 != 0:
        linear = params[-1]
        if n_debye * 2 + 1 < len(params):
            einstein_comps = [
                (params[i], params[i + 1])
                for i in range(n_debye * 2, len(params) - 2, 2)
            ]
    else:
        linear = None
        if n_debye * 2 < len(params):
            einstein_comps = [
                (params[i], params[i + 1])
                for i in range(n_debye * 2, len(params) - 1, 2)
            ]
    return debye_comps, einstein_comps, linear


def create_param_list(debye_comps, einstein_comps, linear):
    """Flatten Debye/Einstein pairs and the linear term into one vector."""
    params = [p for pair in debye_comps for p in pair]
    params += [p for pair in einstein_comps for p in pair]
    if linear is not None:
        params.append(linear)
    return params


def parameter_constrainer(debye_comps, einstein_comps, linear):
    """Rescale prefactors so they sum to 1 and force magnitudes positive."""
    total_prefactors = sum([math.fabs(x[1]) for x in debye_comps + einstein_comps])
    if linear is not None:
        total_prefactors += linear
        linear = math.fabs(linear / total_prefactors)
    einstein_comps = [
        (math.fabs(x[0]), math.fabs(x[1] / total_prefactors)) for x in einstein_comps
    ]
    debye_comps = [
        (math.fabs(x[0]), math.fabs(x[1] / total_prefactors)) for x in debye_comps
    ]
    return debye_comps, einstein_comps, linear


def model_heat_capacity(
    params,
    temps,
    true_values,
    n_debye,
    exponent_val,
    constrain_parameters=True,
    return_totals=False,
):
    """Either the per-component/total curves (return_totals) or the fit error.

    The objective (error) branch is preserved exactly as in the original tool,
    including its normalisation, so the optimiser converges to the same result.
    """
    debye_comps, einstein_comps, linear = param_list_seperate(params, n_debye)
    if constrain_parameters:
        debye_comps, einstein_comps, linear = parameter_constrainer(
            debye_comps, einstein_comps, linear
        )

    debye_ys = [[] for _ in debye_comps]
    einstein_ys = [[] for _ in einstein_comps]
    linear_ys = []
    for T in temps:
        for ys, (Td, Tdp) in zip(debye_ys, debye_comps):
            if Tdp:
                ys.append((Tdp * Debye(T, Td)) / np.power(T, exponent_val))
        for ys, (Te, Tep) in zip(einstein_ys, einstein_comps):
            if Tep:
                ys.append((Tep * Einstein(T, Te)) / np.power(T, exponent_val))
        if linear is not None:
            linear_ys.append((y(linear, T)) / np.power(T, exponent_val))
    all_ys = list(
        zip(*[x for x in debye_ys + einstein_ys + [linear_ys] if len(x) > 0])
    )

    totaly = list(map(sum, all_ys))
    if return_totals:
        return debye_ys, einstein_ys, linear_ys, totaly

    total_err = 0
    for T, yp, yhat in zip(temps, totaly, true_values):
        plot_y = yp / np.power(T, exponent_val)
        plot_yhat = yhat / np.power(T, exponent_val)
        total_err += math.fabs(plot_y - plot_yhat) / math.fabs(max(plot_y, plot_yhat))
    return total_err


class HeatCapacityModel:
    """Model or fit heat-capacity data as a sum of Debye, Einstein and linear terms.

    Ported from the LMDS Flask heat-capacity app. Two Modes share one path:
      ``model``    — fixed components, produce plot + modelled-data CSV.
      ``optimise`` — fit the components to the data first, then model, and also
                     return the fitted components as value outputs.
    """

    def __init__(self):
        self.logger = newLogger("model")

    def process(
        self,
        mode: str,
        values: dict,
        files: dict[str, tuple[Path, str]],
        output_dir: Path,
        parameters: dict,
        logger,
    ) -> dict:
        if mode not in ("model", "optimise"):
            raise ValueError(f"Unknown mode: {mode!r}")

        if "data" not in files:
            raise ValueError("No input data file provided")

        data_path, _ = files["data"]
        temps_data, cp_data = self._read_data(data_path)
        if not temps_data:
            raise ValueError("No data rows found in input file")

        debye_comps = self._to_pairs(values.get("debye_components", []))
        einstein_comps = self._to_pairs(values.get("einstein_components", []))
        linear_in = values.get("linear_component", 0)
        linear = linear_in if linear_in else None  # 0/None -> no linear term

        start_t = float(values["start_temp"])
        end_t = float(values["end_temp"])
        exponent_val = parameters.get("exponent", 1)
        log_x = bool(parameters.get("log_x", False))
        log_y = bool(parameters.get("log_y", False))

        warnings = self._input_warnings(debye_comps, einstein_comps, start_t)
        for w in warnings:
            logger.warning("input warning", detail=w)

        optimise = mode == "optimise"
        result = self._model_and_plot(
            debye_comps, einstein_comps, linear, temps_data, cp_data,
            start_t, end_t, log_x, log_y, exponent_val, output_dir, logger,
            optimise=optimise,
        )

        # Asset outputs (plot.png, model.csv) are written to output_dir and
        # merged in by the runner. Value outputs are returned here.
        outputs: dict = {"warnings": warnings}
        if optimise:
            fitted_debye, fitted_einstein, fitted_linear = result
            outputs["fitted_debye_components"] = self._to_objects(fitted_debye)
            outputs["fitted_einstein_components"] = self._to_objects(fitted_einstein)
            outputs["fitted_linear"] = (
                round(fitted_linear, 5) if fitted_linear is not None else None
            )
        return outputs

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _read_data(path: Path) -> tuple[list[float], list[float]]:
        """Read headerless 'T,Cp' rows (matching the original line parser)."""
        temps, cps = [], []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                temps.append(float(parts[0].rstrip()))
                cps.append(float(parts[1].rstrip()))
        return temps, cps

    @staticmethod
    def _to_pairs(comps: list) -> list[tuple[float, float]]:
        """Accept [{'component':c,'prefactor':p}, ...] -> [(c, p), ...]."""
        pairs = []
        for c in comps:
            if isinstance(c, dict):
                pairs.append((float(c["component"]), float(c["prefactor"])))
            else:  # tolerate [c, p] list form too
                pairs.append((float(c[0]), float(c[1])))
        return pairs

    @staticmethod
    def _to_objects(pairs: list) -> list[dict]:
        return [
            {"component": round(c, 3), "prefactor": round(p, 3)} for c, p in pairs
        ]

    @staticmethod
    def _input_warnings(debye_comps, einstein_comps, start_t) -> list[str]:
        warnings: list[str] = []
        prefactor_sum = sum(x[1] for x in einstein_comps + debye_comps)
        if debye_comps or einstein_comps:
            if prefactor_sum != 1:
                warnings.append(
                    "The Debye and Einstein prefactors do not sum to 1; they have "
                    "been rescaled accordingly."
                )
        if start_t < 1.7:
            warnings.append(
                "This model does not work below approximately 1.6 Kelvin."
            )
        return warnings

    def _model_and_plot(
        self, debye_comps, einstein_comps, linear, data_0, data_1,
        start_t, end_t, log_x, log_y, exponent_val, output_dir, logger,
        optimise=False,
    ):
        temps = np.arange(start_t, end_t, 0.1)
        params = create_param_list(debye_comps, einstein_comps, linear)
        logger.debug("params", params=params)

        if optimise:
            bounds = [
                (1.75, None) if i % 2 == 0 else (0, 1) for i in range(len(params))
            ]
            if linear is not None:
                bounds[-1] = (0, 1)
            res = minimize(
                model_heat_capacity, params,
                args=(data_0, data_1, len(debye_comps), exponent_val),
                bounds=bounds,
            )
            if not res.success:
                logger.error("optimisation failed", message=res.message)
                raise RuntimeError(f"Failed to optimise: {res.message}")
            params = res.x
            logger.debug("params optimised", params=list(params))
            debye_comps, einstein_comps, linear = parameter_constrainer(
                *param_list_seperate(params, len(debye_comps))
            )
            params = create_param_list(debye_comps, einstein_comps, linear)

        debye_ys, einstein_ys, linear_ys, totaly = model_heat_capacity(
            params, temps, data_1, len(debye_comps), exponent_val,
            constrain_parameters=False, return_totals=True,
        )

        # --- plot ---
        plt.scatter(
            data_0,
            [data_1[i] / data_0[i] ** int(exponent_val) for i in range(len(data_0))],
            c="r", s=10, label="data",
        )
        plt.plot(temps, totaly, c="black", label="total")
        data_dict = {"T": temps, "Total": totaly}
        for i, ys in enumerate(debye_ys):
            if len(ys) > 0:
                data_dict[f"D{i + 1}"] = ys
                plt.plot(temps, ys, label=f"D{i + 1}")
        for i, ys in enumerate(einstein_ys):
            if len(ys) > 0:
                data_dict[f"E{i + 1}"] = ys
                plt.plot(temps, ys, label=f"E{i + 1}")
        if linear is not None:
            data_dict["Linear"] = linear_ys
            plt.plot(temps, linear_ys, label="Linear")

        plt.xlabel("$T$")
        if exponent_val > 1:
            plt.ylabel(f"$C_p/T^{exponent_val}$")
        elif exponent_val == 1:
            plt.ylabel("$C_p/T$")
        elif exponent_val == 0:
            plt.ylabel("$C_p$")
        if log_x:
            plt.xlabel("$log T$")
            plt.xscale("log")
        if log_y:
            plt.yscale("log")
            if exponent_val > 1:
                plt.ylabel(f"$log(C_p/T^{exponent_val})$")
            elif exponent_val == 1:
                plt.ylabel("$log(C_p/T)$")
            elif exponent_val == 0:
                plt.ylabel("$log(C_p)$")
        plt.legend()
        plt.tight_layout()

        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "plot.png")
        plt.close()

        pd.DataFrame(data_dict).to_csv(output_dir / "model.csv", index=False)
        logger.info("wrote assets", plot="plot.png", data="model.csv")

        if optimise:
            return debye_comps, einstein_comps, linear
        return None

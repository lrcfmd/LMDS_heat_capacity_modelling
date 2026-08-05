from model.model import HeatCapacityModel

# Wire the concrete model for the runner. This is the single tool-owned wiring
# point — the runner imports `handler` from here.
handler = HeatCapacityModel()

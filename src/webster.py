"""
webster.py
----------
Reusable Webster's method calculation, called on demand by
simulation.py's TrafficAnalyzer for a single (area, road) at a time.

This is NOT the same as webster_preprocessing.py + webster_batch.py,
which run the same underlying math as a batch over the whole dataset
and write webster_input_full.csv / webster_results_full.csv for
reporting. Both use the same core formula (C0 = (1.5L+5)/(1-Y)); this
version just takes averaged traffic numbers for one location and
returns a result object instead of writing a CSV.

PROJECT ASSUMPTION: traffic demand is split evenly across all phases,
since the dataset has no per-phase turning-movement data. Saturation
flow is derived purely from intersection geometry (num_phases x
lanes_per_phase), which the caller supplies -- not estimated from
traffic volume like the batch pipeline does.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

SATURATION_FLOW_PER_LANE = 1800
LOST_TIME_PER_PHASE = 4
MAX_PRACTICAL_CYCLE = 180

def _baseline_cycle_from_utilization(utilization_pct: float) -> int:
    if utilization_pct >= 100:
        return 120
    elif utilization_pct >= 90:
        return 100
    elif utilization_pct >= 75:
        return 80
    else:
        return 60

@dataclass
class SignalTimingResult:
    traffic_volume_vph: float
    road_capacity_utilization_pct: float
    num_phases: int
    lanes_per_phase: int
    saturation_flow_total: float
    flow_ratio_per_phase: float
    total_flow_ratio_y: float
    webster_status: str
    lost_time_seconds: int
    baseline_cycle_seconds: int
    optimal_cycle_seconds: float
    cycle_capped: bool
    available_green_seconds: float
    green_per_phase_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)

def estimate_signal_timing(
    traffic_volume_vph: float,
    road_capacity_utilization_pct: float,
    num_phases: int = 4,
    lanes_per_phase: int = 2,
) -> SignalTimingResult:

    if num_phases < 2:
        raise ValueError("num_phases must be at least 2")
    if lanes_per_phase < 1:
        raise ValueError("lanes_per_phase must be at least 1")

    saturation_flow_per_phase = lanes_per_phase * SATURATION_FLOW_PER_LANE
    saturation_flow_total = num_phases * saturation_flow_per_phase

    phase_flow = traffic_volume_vph / num_phases
    flow_ratio_per_phase = (
        phase_flow / saturation_flow_per_phase if saturation_flow_per_phase > 0 else 0
    )
    total_y = round(flow_ratio_per_phase * num_phases, 4)

    lost_time = num_phases * LOST_TIME_PER_PHASE
    baseline_cycle = _baseline_cycle_from_utilization(road_capacity_utilization_pct)

    if total_y <= 0:
        webster_status = "NO_VALID_DEMAND"
    elif total_y >= 1:
        webster_status = "OVERSATURATED"
    else:
        webster_status = "VALID"

    if webster_status == "VALID":
        raw_cycle = (1.5 * lost_time + 5) / (1 - total_y)
        cycle_capped = raw_cycle > MAX_PRACTICAL_CYCLE
        optimal_cycle = round(min(raw_cycle, MAX_PRACTICAL_CYCLE), 2)
        available_green = round(optimal_cycle - lost_time, 2)
        green_per_phase = round(available_green / num_phases, 2)
    else:
        cycle_capped = False
        optimal_cycle = float("nan")
        available_green = float("nan")
        green_per_phase = float("nan")

    return SignalTimingResult(
        traffic_volume_vph=round(traffic_volume_vph, 1),
        road_capacity_utilization_pct=round(road_capacity_utilization_pct, 1),
        num_phases=num_phases,
        lanes_per_phase=lanes_per_phase,
        saturation_flow_total=saturation_flow_total,
        flow_ratio_per_phase=round(flow_ratio_per_phase, 4),
        total_flow_ratio_y=total_y,
        webster_status=webster_status,
        lost_time_seconds=lost_time,
        baseline_cycle_seconds=baseline_cycle,
        optimal_cycle_seconds=optimal_cycle,
        cycle_capped=cycle_capped,
        available_green_seconds=available_green,
        green_per_phase_seconds=green_per_phase,
    )
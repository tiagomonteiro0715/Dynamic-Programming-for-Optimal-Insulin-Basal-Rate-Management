"""
 • Final Project: CS 5800
 • Dynamic Programming for Optimal Insulin Basal Rate Management
 • Muhammed Bilal
 • Tiago Monteiro

 DATA DISCLOSURE:
 The hourly glucose means used in this project are aggregate statistics
 derived from de-identified patient data. No individual patient records
 or personally identifiable health information (PHI) are included.
 All data has been aggregated and anonymized in compliance with privacy regulations.
"""

import numpy as np
from typing import List, Tuple, Dict
import math

# Pump micro-steps (U per 5 minutes)
U5_OPTIONS = [round(x * 0.05, 2) for x in range(0, 9)]  # 0.00, 0.05, ..., 0.40

# Convert helpers
def u5_to_u_per_hr(u5: float) -> float:
    return u5 * 12.0

def u_per_hr_to_u5(u_hr: float) -> float:
    return u_hr / 12.0

# Simulation parameters
SEGMENT_DURATION_HR = 0.5  # 30 minutes
N_SEGMENTS = 12            # 6 hours / 0.5 hr
TARGET_G = 6.0             # mmol/L target
HYPO_THRESHOLD = 3.9       # mmol/L
INSULIN_SENSITIVITY = 2.0  # mmol/L drop per unit (my pump setting)

# Cost weights
W_DEV = 1.0       # weight for squared deviation from target (per mmol/L^2)
W_HYPO = 30.0     # weight for cubic hypoglycemia penalty
W_SMOOTH = 0.2    # weight for squared change in U/5min (smoothness)

# Max allowed change between adjacent 30-min segments (in U/5min)
MAX_DELTA_U5 = 0.10  

def glucose_deviation_cost(g: float, target: float = TARGET_G) -> float:
    """Quadratic penalty for deviation from target."""
    return W_DEV * (g - target) ** 2

def hypoglycemia_cost(g: float, hypo_threshold: float = HYPO_THRESHOLD) -> float:
    """Cubic penalty for being below hypo threshold."""
    if g >= hypo_threshold:
        return 0.0
    diff = hypo_threshold - g
    return W_HYPO * (diff ** 3)

def smoothness_cost(prev_u5: float, new_u5: float) -> float:
    """Penalty for abrupt changes in basal (U/5min units)."""
    delta = abs(new_u5 - prev_u5)
    return W_SMOOTH * (delta ** 2)

def total_transition_cost(glucose: float, prev_u5: float, new_u5: float) -> float:
    """Total immediate cost after applying new_u5 for the segment (evaluated on resulting glucose)."""
    return glucose_deviation_cost(glucose) + hypoglycemia_cost(glucose) + smoothness_cost(prev_u5, new_u5)

def apply_insulin_effect(current_g: float, basal_u5: float, duration_hr: float = SEGMENT_DURATION_HR) -> float:
    """
    Simple instantaneous model: insulin delivered in the segment reduces glucose proportionally.
    units_delivered = (U/5min) * (60/5) * duration_hr / (60/5) = basal_u5 * (duration_hr * 12)
    But simpler: units_delivered = basal_u5 * (duration_hr * 12) / 6? No — correct:
    basal_u5 is units per 5 minutes. In duration_hr hours, number of 5-min intervals = duration_hr * 12.
    units_delivered = basal_u5 * (duration_hr * 12)
    glucose drop = INSULIN_SENSITIVITY * units_delivered
    """
    intervals = duration_hr * 12.0
    units = basal_u5 * intervals
    glucose_drop = INSULIN_SENSITIVITY * units
    return current_g - glucose_drop

HOURLY_MEANS_12H = np.array([
    8.2,  # 12am-1am
    9.3,  # 1-2
    10.7, # 2-3
    11.0, # 3-4
    10.8, # 4-5
    11.0, # 5-6
    10.3, # 6-7
    8.9,  # 7-8
    8.0,  # 8-9
    7.4,  # 9-10
    7.8,  # 10-11
    8.3   # 11-12
])

def build_baseline_segments(hourly_means_12h: np.ndarray,
                            start_hour_index: int = 3,  # index for 3am in the array above
                            window_hours: float = 6.0,
                            n_segments: int = N_SEGMENTS) -> np.ndarray:
    """
    Extract the relevant hourly window (3am-9am) and upsample/interpolate to n_segments.
    start_hour_index = 3 corresponds to 3am in HOURLY_MEANS_12H.
    """
    indices = np.arange(start_hour_index, start_hour_index + 7)  # 3..9
    hours = np.linspace(0, 6, len(indices))  # 7 points across 6 hours
    target_times = np.linspace(0, 6, n_segments)  # 12 segments across 6 hours
    values = hourly_means_12h[indices]
    baseline_segments = np.interp(target_times, hours, values)
    return baseline_segments  # length n_segments

# Top-Down DP (memoization)
class TopDownDP:
    def __init__(self,
                 baseline_segments: np.ndarray,
                 possible_u5: List[float],
                 insulin_sensitivity: float = INSULIN_SENSITIVITY):
        self.baseline = baseline_segments
        self.n = len(baseline_segments)
        self.possible_u5 = possible_u5
        self.insulin_sensitivity = insulin_sensitivity
        self.memo: Dict[Tuple[int, float, float], Tuple[float, List[float]]] = {}

    def solve(self, seg: int, prev_u5: float, current_g: float) -> Tuple[float, List[float]]:
        """
        Returns (min_cost, best_path_of_u5s_from_seg_to_end)
        Memoization key discretizes glucose to 1 decimal to keep state space manageable.
        """
        if seg == self.n:
            return 0.0, []

        key = (seg, round(prev_u5, 3), round(current_g, 1))
        if key in self.memo:
            return self.memo[key]

        best_cost = math.inf
        best_path: List[float] = []

        for new_u5 in self.possible_u5:
            # enforce max delta constraint
            if abs(new_u5 - prev_u5) > MAX_DELTA_U5:
                continue

            # Predict glucose after applying new_u5 for this segment
            next_g = apply_insulin_effect(current_g, new_u5, SEGMENT_DURATION_HR)

            # Add baseline drift: small nudging toward baseline segment value
            # This models non-insulin drivers (hormones, circadian) as a fraction of difference
            baseline_target = self.baseline[seg]
            drift = 0.1 * (baseline_target - current_g)  # 10% of gap per segment
            next_g += drift

            # Immediate cost evaluated on next_g
            immediate_cost = total_transition_cost(next_g, prev_u5, new_u5)

            # Recurse
            future_cost, future_path = self.solve(seg + 1, new_u5, next_g)
            total_cost = immediate_cost + future_cost

            if total_cost < best_cost:
                best_cost = total_cost
                best_path = [new_u5] + future_path

        # If no feasible new_u5 (shouldn't happen), allow staying same rate
        if best_cost == math.inf:
            next_g = apply_insulin_effect(current_g, prev_u5, SEGMENT_DURATION_HR)
            baseline_target = self.baseline[seg]
            next_g += 0.1 * (baseline_target - current_g)
            immediate_cost = total_transition_cost(next_g, prev_u5, prev_u5)
            future_cost, future_path = self.solve(seg + 1, prev_u5, next_g)
            best_cost = immediate_cost + future_cost
            best_path = [prev_u5] + future_path

        self.memo[key] = (best_cost, best_path)
        return self.memo[key]

    def get_optimal(self, initial_g: float, initial_u5: float) -> Tuple[float, List[float]]:
        self.memo.clear()
        return self.solve(0, initial_u5, initial_g)

# ---------------------------
# Bottom-Up DP (tabulation)
# ---------------------------

class BottomUpDP:
    def __init__(self,
                 baseline_segments: np.ndarray,
                 possible_u5: List[float],
                 insulin_sensitivity: float = INSULIN_SENSITIVITY):
        self.baseline = baseline_segments
        self.n = len(baseline_segments)
        self.possible_u5 = possible_u5
        self.n_rates = len(possible_u5)
        self.insulin_sensitivity = insulin_sensitivity

        # Discretize glucose for tractable DP table
        self.glucose_levels = np.round(np.linspace(2.0, 16.0, 141), 2)  # 2.0 to 16.0 mmol/L step 0.1
        self.n_glucose = len(self.glucose_levels)

    def discretize_glucose_idx(self, g: float) -> int:
        return int(np.argmin(np.abs(self.glucose_levels - g)))

    def solve(self, initial_g: float, initial_u5: float) -> Tuple[float, List[float]]:
        dp = np.full((self.n + 1, self.n_rates, self.n_glucose), np.inf)
        parent = {} 

        init_g_idx = self.discretize_glucose_idx(initial_g)
        try:
            init_rate_idx = self.possible_u5.index(initial_u5)
        except ValueError:
            init_rate_idx = int(np.argmin([abs(r - initial_u5) for r in self.possible_u5]))

        dp[0, init_rate_idx, init_g_idx] = 0.0

        for seg in range(self.n):
            for prev_rate_idx, prev_u5 in enumerate(self.possible_u5):
                for g_idx, g_val in enumerate(self.glucose_levels):
                    if dp[seg, prev_rate_idx, g_idx] == np.inf:
                        continue
                    current_cost = dp[seg, prev_rate_idx, g_idx]
                    current_g = g_val

                    for new_rate_idx, new_u5 in enumerate(self.possible_u5):
                        # enforce max delta
                        if abs(new_u5 - prev_u5) > MAX_DELTA_U5:
                            continue

                        # simulate next glucose
                        next_g = apply_insulin_effect(current_g, new_u5, SEGMENT_DURATION_HR)
                        baseline_target = self.baseline[seg]
                        next_g += 0.1 * (baseline_target - current_g)
                        next_g_idx = self.discretize_glucose_idx(next_g)

                        # transition cost
                        t_cost = total_transition_cost(self.glucose_levels[next_g_idx], prev_u5, new_u5)
                        new_cost = current_cost + t_cost

                        if new_cost < dp[seg + 1, new_rate_idx, next_g_idx]:
                            dp[seg + 1, new_rate_idx, next_g_idx] = new_cost
                            parent[(seg + 1, new_rate_idx, next_g_idx)] = (seg, prev_rate_idx, g_idx, new_u5)

        # find minimal cost at final segment
        final_slice = dp[self.n]
        min_idx = np.unravel_index(np.argmin(final_slice), final_slice.shape)
        min_cost = final_slice[min_idx]
        optimal_rates: List[float] = []
        cur_state = (self.n, min_idx[0], min_idx[1])
        while cur_state in parent:
            seg, rate_idx, g_idx, chosen_u5 = parent[cur_state]
            optimal_rates.insert(0, chosen_u5)
            cur_state = (seg, rate_idx, g_idx)
        # If path shorter than n, pad with initial rate
        if len(optimal_rates) < self.n:
            pad = [self.possible_u5[init_rate_idx]] * (self.n - len(optimal_rates))
            optimal_rates = pad + optimal_rates
        return float(min_cost), optimal_rates

# ---------------------------
# Simulation helpers & metrics
# ---------------------------

def simulate_glucose_trajectory(initial_g: float, initial_u5: float, schedule_u5: List[float],
                                baseline_segments: np.ndarray) -> List[float]:
    """Simulate glucose values at the end of each segment given a schedule of U/5min rates."""
    g = initial_g
    trajectory = []
    prev_u5 = initial_u5
    for seg, u5 in enumerate(schedule_u5):
        # apply insulin effect
        g = apply_insulin_effect(g, u5, SEGMENT_DURATION_HR)
        # baseline drift
        g += 0.1 * (baseline_segments[seg] - g)
        trajectory.append(g)
        prev_u5 = u5
    return trajectory

def compute_metrics(trajectory: List[float], schedule_u5: List[float]) -> Dict[str, float]:
    arr = np.array(trajectory)
    mean_g = float(np.mean(arr))
    std_g = float(np.std(arr))
    time_in_range = float(np.mean((arr >= 3.9) & (arr <= 10.0))) * 100.0  # percent
    total_insulin_units = sum([u5 * (SEGMENT_DURATION_HR * 12.0) for u5 in schedule_u5])  # units delivered
    hypo_events = float(np.sum(arr < 3.9))
    return {
        "mean_glucose_mmol_L": mean_g,
        "std_glucose_mmol_L": std_g,
        "time_in_range_pct": time_in_range,
        "total_insulin_units": total_insulin_units,
        "hypo_count": hypo_events
    }

# ---------------------------
# Main example run
# ---------------------------

def main():
    # Build baseline segments from hourly means (3am-9am)
    baseline_segments = build_baseline_segments(HOURLY_MEANS_12H, start_hour_index=3, n_segments=N_SEGMENTS)

    initial_glucose = 10.5  # mmol/L (example high starting BG)
    initial_u5 = 0.10       # U per 5 minutes (current pump rate at start)

    print("\n=== Baseline segments (3:00am -> 9:00am) ===")
    print(np.round(baseline_segments, 2))

    print("\nPossible pump micro-steps (U per 5min):", U5_OPTIONS)
    print(f"Initial glucose: {initial_glucose} mmol/L, initial pump rate: {initial_u5} U/5min")

    # Instantiate solvers
    td_solver = TopDownDP(baseline_segments, U5_OPTIONS)
    bu_solver = BottomUpDP(baseline_segments, U5_OPTIONS)

    # Top-Down DP
    td_cost, td_schedule = td_solver.get_optimal(initial_glucose, initial_u5)
    td_traj = simulate_glucose_trajectory(initial_glucose, initial_u5, td_schedule, baseline_segments)
    td_metrics = compute_metrics(td_traj, td_schedule)

    print("\n Top-Down DP Result ")
    print("Total cost (internal):", round(td_cost, 2))
    print("Schedule (U/5min):", [round(x, 3) for x in td_schedule])
    print("Simulated glucose trajectory (end of each 30-min):", [round(x, 2) for x in td_traj])
    print("Metrics:", {k: round(v, 3) for k, v in td_metrics.items()})
    print("Memoization states cached:", len(td_solver.memo))

    # Bottom-Up DP
    bu_cost, bu_schedule = bu_solver.solve(initial_glucose, initial_u5)
    bu_traj = simulate_glucose_trajectory(initial_glucose, initial_u5, bu_schedule, baseline_segments)
    bu_metrics = compute_metrics(bu_traj, bu_schedule)

    print("\n Bottom-Up DP Result")
    print("Total cost (internal):", round(bu_cost, 2))
    print("Schedule (U/5min):", [round(x, 3) for x in bu_schedule])
    print("Simulated glucose trajectory (end of each 30-min):", [round(x, 2) for x in bu_traj])
    print("Metrics:", {k: round(v, 3) for k, v in bu_metrics.items()})

if __name__ == "__main__":
    main()
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
from typing import List, Tuple
import time

from dynamic_programming import (
    U5_OPTIONS,
    SEGMENT_DURATION_HR,
    N_SEGMENTS,
    MAX_DELTA_U5,
    HOURLY_MEANS_12H,
    apply_insulin_effect,
    total_transition_cost,
    build_baseline_segments,
    simulate_glucose_trajectory,
    compute_metrics
)


class MyopicGreedy:
    """
    Single-step lookahead greedy algorithm.
    At each segment, choose the basal rate that minimizes immediate cost.
    """
    def __init__(self, baseline_segments: np.ndarray, possible_u5: List[float]):
        self.baseline = baseline_segments
        self.n = len(baseline_segments)
        self.possible_u5 = possible_u5

    def solve(self, initial_g: float, initial_u5: float) -> Tuple[float, List[float]]:
        schedule = []
        g = initial_g
        prev_u5 = initial_u5
        total_cost = 0.0
        
        for seg in range(self.n):
            best_u5 = prev_u5
            best_cost = float('inf')
            
            for new_u5 in self.possible_u5:
                if abs(new_u5 - prev_u5) > MAX_DELTA_U5:
                    continue
                
                # Simulate next glucose
                next_g = apply_insulin_effect(g, new_u5, SEGMENT_DURATION_HR)
                next_g += 0.1 * (self.baseline[seg] - g)  # baseline drift
                
                cost = total_transition_cost(next_g, prev_u5, new_u5)
                
                if cost < best_cost:
                    best_cost = cost
                    best_u5 = new_u5
            
            schedule.append(best_u5)
            total_cost += best_cost
            
            # Update state
            g = apply_insulin_effect(g, best_u5, SEGMENT_DURATION_HR)
            g += 0.1 * (self.baseline[seg] - g)
            prev_u5 = best_u5
        
        return total_cost, schedule



def main():
    baseline_segments = build_baseline_segments(HOURLY_MEANS_12H, start_hour_index=3, n_segments=N_SEGMENTS)

    # Starting conditions
    initial_glucose = 10.5  
    initial_u5 = 0.10       

    print("\n=== Myopic Greedy Basal Optimization ===")
    print(f"Initial glucose: {initial_glucose} mmol/L")
    print(f"Initial pump rate: {initial_u5} U/5min")
    print(f"Baseline segments (3am-9am): {np.round(baseline_segments, 2)}")

    greedy_solver = MyopicGreedy(baseline_segments, U5_OPTIONS)
    
    start = time.time()
    greedy_cost, greedy_schedule = greedy_solver.solve(initial_glucose, initial_u5)
    greedy_time = time.time() - start
    
    greedy_traj = simulate_glucose_trajectory(initial_glucose, initial_u5, greedy_schedule, baseline_segments)
    greedy_metrics = compute_metrics(greedy_traj, greedy_schedule)
    
    print(f"Computation time: {greedy_time*1000:.2f} ms")
    print(f"Total cost: {greedy_cost:.2f}")
    print(f"Schedule (U/5min): {[round(x, 3) for x in greedy_schedule]}")
    print(f"Trajectory: {[round(x, 2) for x in greedy_traj]}")
    print(f"Metrics:")
    for key, value in greedy_metrics.items():
        print(f"  {key}: {round(value, 3)}")


if __name__ == "__main__":
    main()
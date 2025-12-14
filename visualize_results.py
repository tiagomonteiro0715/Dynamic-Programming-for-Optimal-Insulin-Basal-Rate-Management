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

import matplotlib.pyplot as plt
import numpy as np
from dynamic_programming import (
    TopDownDP, BottomUpDP, U5_OPTIONS, N_SEGMENTS, HOURLY_MEANS_12H,
    build_baseline_segments, simulate_glucose_trajectory, TARGET_G, 
    HYPO_THRESHOLD, SEGMENT_DURATION_HR
)
from greedy_approach import MyopicGreedy


def main():
    baseline_segments = build_baseline_segments(HOURLY_MEANS_12H, start_hour_index=3, n_segments=N_SEGMENTS)
    initial_glucose = 10.5
    initial_u5 = 0.10
    
    td_solver = TopDownDP(baseline_segments, U5_OPTIONS)
    td_cost, td_schedule = td_solver.get_optimal(initial_glucose, initial_u5)
    td_traj = simulate_glucose_trajectory(initial_glucose, initial_u5, td_schedule, baseline_segments)
    
    bu_solver = BottomUpDP(baseline_segments, U5_OPTIONS)
    bu_cost, bu_schedule = bu_solver.solve(initial_glucose, initial_u5)
    bu_traj = simulate_glucose_trajectory(initial_glucose, initial_u5, bu_schedule, baseline_segments)
    
    greedy_solver = MyopicGreedy(baseline_segments, U5_OPTIONS)
    greedy_cost, greedy_schedule = greedy_solver.solve(initial_glucose, initial_u5)
    greedy_traj = simulate_glucose_trajectory(initial_glucose, initial_u5, greedy_schedule, baseline_segments)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    time_points = np.arange(0, N_SEGMENTS + 1) * SEGMENT_DURATION_HR
    
    ax.plot(time_points, [initial_glucose] + td_traj, 'o-', label='Top-Down DP', linewidth=2, markersize=6)
    ax.plot(time_points, [initial_glucose] + bu_traj, 's-', label='Bottom-Up DP', linewidth=2, markersize=6)
    ax.plot(time_points, [initial_glucose] + greedy_traj, '^-', label='Greedy', linewidth=2, markersize=6)
    
    ax.axhline(y=TARGET_G, color='green', linestyle=':', linewidth=2, label=f'Target ({TARGET_G} mmol/L)')
    ax.axhline(y=HYPO_THRESHOLD, color='red', linestyle=':', linewidth=2, label=f'Hypo ({HYPO_THRESHOLD} mmol/L)')
    ax.fill_between(time_points, 3.9, 10.0, alpha=0.1, color='green')
    
    ax.set_ylabel('Glucose (mmol/L)', fontsize=12)
    ax.set_title('Algorithm comparison for glucose trajectories', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('glucose_comparison.png', dpi=300)
    plt.show()
    
    print(f"\nTop-Down cost: {td_cost:.2f}")
    print(f"Bottom-Up cost: {bu_cost:.2f}")
    print(f"Greedy cost: {greedy_cost:.2f}")


if __name__ == "__main__":
    main()
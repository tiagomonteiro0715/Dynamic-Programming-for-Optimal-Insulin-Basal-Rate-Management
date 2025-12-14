# Dynamic Programming for Optimal Insulin Basal Rate Management


![Glucose Trajectory Comparison](https://raw.githubusercontent.com/tiagomonteiro0715/Dynamic-Programming-for-Optimal-Insulin-Basal-Rate-Management/main/glucose_comparison.png)


## Final Project Academic Context

**Course**: CS 5800 - Algorithms  
**Institution**: Northeastern University  
**Semester**: Fall 2024

---

## Project Overview

This project implements and compares dynamic programming algorithms for optimizing insulin basal rates in diabetes management. 


The goal is to determine the optimal insulin pump delivery schedule that maintains blood glucose levels within a safe target range while minimizing hypoglycemia risk and pump rate fluctuations.

### Key Features

- **Top-Down Dynamic Programming** (Memoization): Recursive approach with caching
- **Bottom-Up Dynamic Programming** (Tabulation): Iterative table-filling approach  
- **Myopic Greedy Baseline**: Single-step lookahead for comparison
- **Comprehensive Visualization**: Glucose trajectory comparisons across algorithms

---

## Problem Statement

Managing insulin delivery for Type 1 diabetes requires balancing multiple competing objectives:

1. **Maintain target glucose** (~6.0 mmol/L)
2. **Avoid hypoglycemia** (<3.9 mmol/L) 
3. **Minimize pump rate changes** (patient comfort)
4. **Account for circadian patterns** (dawn phenomenon, fasting periods)

This is formulated as a multi-stage optimization problem over a 6-hour window (3am-9am), divided into 12 segments of 30 minutes each.
---

## Installation

### Requirements

```bash
pip install numpy matplotlib
```

### Files Structure

```
project/
├── dynamic_programming.py    # Core DP implementations
├── greedy_approach.py        # Myopic greedy baseline
├── visualize_results.py      # Comparison visualization
└── README.md                 # This file
```

---

## Usage

### Run Individual Algorithms

**Top-Down & Bottom-Up DP:**
```bash
python dynamic_programming.py
```

**Greedy Approach:**
```bash
python greedy_approach.py
```

### Generate Comparison Visualization

```bash
python visualize_results.py
```

This creates `glucose_comparison.png` showing all three algorithms' glucose trajectories.

---

## Limitations & Future Work

### Current Limitations

- Simplified physiological model (no insulin absorption delay, digestion, exercise)
- Deterministic dynamics (no stochastic glucose variability)
- Fixed insulin sensitivity (varies by individual and time of day)

### Future work

1. **Model Predictive Control**: Receding horizon optimization with real-time updates
2. **Reinforcement Learning**: Learn optimal policies from patient-specific data

---

## References

### Dynamic Programming
- Cormen, T. H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
- Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.

---

## License

This project is for academic purposes only. Not intended for clinical use.

## Contact

For questions about this project:
- Muhammed Bilal: bilal.m@northeastern.edu
- Tiago Monteiro: monteiro.t@northeastern.edu 

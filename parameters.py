import numpy as np


scan_range = {
    'lethality': np.arange(0.005, 0.021, 0.001),
    'burn-in': np.arange(12, 23, 1),
    't-death': np.arange(14, 32, 2),
    't-confirmed': [7],
    'infected-start': np.arange(20, 105, 5),
    'detection-rate': np.arange(0.025, 0.225, 0.025),
    'R0-0': np.arange(2., 3.1, 0.1),
    'day-action-1': np.arange(10, 22, 2),
    'R0-1': np.arange(0.5, 2.1, 0.1),
    'R0-2': np.arange(0.5, 2.1, 0.1),
    'R0-lo-A': np.arange(0.1, 2.1, 0.1),      # R0 convergence for t -> infinity
    'R0-lo-B': np.arange(10, 30, 2),          # time t at reversal point of R0
    'R0-lo-C': np.logspace(-1, 1, 10)       # time scale for transition of R0 from R0-A to R0-B
}

digits = {
    'lethality': 3,
    'burn-in': 0,
    't-death': 0,
    'detection-rate': 2,
    'R0-0': 1,
    'day-action-1': 1,
    'R0-1': 1,
    'day-action-2': 1,
    'R0-2': 1,
    'R0-lo-A': 1,
    'R0-lo-B': 1,
    'R0-lo-C': 1,
    'infected-start': 0,
    't-confirmed': 0,
}

defaults = {
    'attack-rate': 0.15,
    'R0-0': 2.9,
    'detection-rate': 0.07,
    'lethality': 0.004,
    'burn-in': 14,
    'day-action-1': 14,
    'day-action-2': 21,
    't-contagious': 4,
    't-cured': 14,
    't-death': 12,
    't-confirmed': 7,
    'infected-start': 90,
    'contagious-start': 0,
    'confirmed-start': 0
}

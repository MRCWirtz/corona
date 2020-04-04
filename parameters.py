import numpy as np


scan_range = {
    'lethality': np.arange(0.005, 0.021, 0.001),
    'burn-in': np.arange(8, 21, 1),
    't-death': np.arange(14, 32, 2),
    't-confirmed': [7],
    'infected-start': np.arange(20, 85, 5),
    'detection-rate': np.arange(0.1, 1., 0.1),
    'R0-0': np.arange(2., 3.2, 0.1),
    'R0-1': np.arange(0.1, 2.1, 0.1),
    'R0-2': np.arange(0.1, 2.1, 0.1),
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
    'R0-1': 1,
    'R0-2': 1,
    'R0-lo-A': 1,
    'R0-lo-B': 1,
    'R0-lo-C': 1,
    'infected-start': 0,
    't-confirmed': 0,
}

defaults = {
    'attack-rate': 0.15,
    'R0-0': 2.6,
    'detection-rate': 0.33,
    'lethality': 0.015,
    'burn-in': 13,
    'day-action-1': 17,
    'day-action-2': 24,
    't-contagious': 4,
    't-cured': 14,
    't-death': 12,
    't-confirmed': 7,
    'infected-start': 45,
    'contagious-start': 0,
    'confirmed-start': 0
}

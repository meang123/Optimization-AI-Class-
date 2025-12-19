#!/usr/bin/env python
# Analysis script for ppo_ant_adam experiment

from utils.plotter import Plotter
from utils.sweeper import unfinished_index, time_info, memory_info
import numpy as np
from scipy.stats import bootstrap
from collections import namedtuple


def get_process_result_dict(result, config_idx, mode='Test'):
  result_dict = {
    'Env': result['Env'][0],
    'Agent': result['Agent'][0],
    'Config Index': config_idx,
    'Return (mean)': result['Return'][-100:].mean(skipna=False) if mode=='Train' else result['Return'][-2:].mean(skipna=False)
  }
  return result_dict


def get_csv_result_dict(result, config_idx, mode='Train', ci=90, method='percentile'):
  return_mean = result['Return (mean)'].values.tolist()
  if len(return_mean) > 1:
    CI = bootstrap(
      (result['Return (mean)'].values.tolist(),),
      np.mean, confidence_level=ci/100,
      method=method
    ).confidence_interval
  else:
    CI = namedtuple('ConfidenceInterval', ['low', 'high'])(low=return_mean[0], high=return_mean[0])
  result_dict = {
    'Env': result['Env'][0],
    'Agent': result['Agent'][0],
    'Config Index': config_idx,
    'Return (mean)': result['Return (mean)'].mean(skipna=False),
    'Return (se)': result['Return (mean)'].sem(ddof=0),
    'Return (bootstrap_mean)': (CI.high + CI.low) / 2,
    f'Return (ci={ci})': (CI.high - CI.low) / 2,
  }
  return result_dict


def analyze(exp, runs=1):
  cfg = {
    'exp': exp,
    'merged': True,
    'x_label': 'Step',
    'y_label': 'Return',
    'rolling_score_window': -1,
    'hue_label': 'Agent',
    'show': False,
    'imgType': 'png',
    'estimator': 'mean',
    'ci': ('ci', 90),
    'x_format': None,
    'y_format': None,
    'xlim': {'min': None, 'max': None},
    'ylim': {'min': None, 'max': None},
    'EMA': True,
    'loc': 'upper left',
    'sweep_keys': ['optim/name', 'optim/kwargs/learning_rate'],
    'sort_by': ['Return (mean)', 'Return (se)'],
    'ascending': [False, True],
    'runs': runs
  }

  plotter = Plotter(cfg)
  plotter.sweep_keys = ['optim/name', 'optim/kwargs/learning_rate']

  mode = 'Test'
  plotter.csv_merged_results(mode, get_csv_result_dict, get_process_result_dict)
  plotter.plot_results(mode, indexes='all')


if __name__ == "__main__":
  exp_name = 'ppo_ant_adam'
  runs = 1  # Number of runs (seeds) for this experiment

  print(f"\n{'='*60}")
  print(f"Analyzing experiment: {exp_name}")
  print(f"{'='*60}\n")

  print("1. Checking for unfinished jobs...")
  unfinished_index(exp_name, runs=runs)

  print("\n2. Analyzing memory usage...")
  memory_info(exp_name, runs=runs)

  print("\n3. Analyzing runtime...")
  time_info(exp_name, runs=runs)

  print("\n4. Generating CSV results and plots...")
  analyze(exp_name, runs=runs)

  print(f"\n{'='*60}")
  print("Analysis complete!")
  print(f"Results saved to: logs/{exp_name}/")
  print(f"{'='*60}\n")

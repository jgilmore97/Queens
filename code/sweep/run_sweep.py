import argparse
import json
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import NopPruner
from pathlib import Path
from datetime import datetime
from functools import partial

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sweep.objective import objective


def save_results(study: optuna.Study, output_path: str):
    results = []

    for trial in study.trials:
        result = {
            'trial_id': trial.number,
            'state': trial.state.name,
            'value': trial.value,
            # Hyperparameters
            **trial.params,
            # User attributes (solve rates, status)
            **trial.user_attrs,
        }
        results.append(result)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to {output_path}")


def print_summary(study: optuna.Study):
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    print("\n" + "=" * 60)
    print("HYPERPARAMETER SWEEP SUMMARY")
    print("=" * 60)
    print(f"Total trials: {len(study.trials)}")
    print(f"  Completed: {len(completed)}")
    print(f"  Pruned: {len(pruned)}")
    print(f"  Failed: {len(failed)}")

    if completed:
        best_trial = study.best_trial
        print(f"\nBest trial: #{best_trial.number}")
        print(f"  Solve rate: {best_trial.value:.4f}")
        print(f"  Parameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        # Top 5 trials
        sorted_trials = sorted(completed, key=lambda t: t.value or 0, reverse=True)
        print(f"\nTop 5 trials:")
        for i, trial in enumerate(sorted_trials[:5], 1):
            print(f"  {i}. Trial #{trial.number}: solve_rate={trial.value:.4f}")


def main():
    parser = argparse.ArgumentParser(description="HRM Hyperparameter Sweep")
    parser.add_argument('--n_trials', type=int, default=60,
                        help='Number of trials to run')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--train_json', type=str, default='data/StateTrainingSet.json',
                        help='Path to training data')
    parser.add_argument('--state0_json', type=str, default='data/State0TrainingSet.json',
                        help='Path to state-0 training data')
    parser.add_argument('--val_json', type=str, default='data/StateValSet.json',
                        help='Path to validation data')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (default: sweep_results_<timestamp>.json)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--study_name', type=str, default='hrm_sweep',
                        help='Optuna study name')
    args = parser.parse_args()

    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'sweep_results_{timestamp}.json'

    print("=" * 60)
    print("HRM HYPERPARAMETER SWEEP")
    print("=" * 60)
    print(f"Trials: {args.n_trials}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output}")
    print(f"Train data: {args.train_json}")
    print(f"State-0 data: {args.state0_json}")
    print(f"Val data: {args.val_json}")
    print("=" * 60)

    sampler = TPESampler(seed=args.seed)
    pruner = NopPruner() # No pruning for now - compromised result generalizability

    study = optuna.create_study(
        study_name=args.study_name,
        direction='maximize',
        sampler=sampler,
        pruner=pruner
    )

    objective_fn = partial(
        objective,
        train_json=args.train_json,
        state0_json=args.state0_json,
        val_json=args.val_json,
        device=args.device,
        seed=args.seed
    )

    study.optimize(
        objective_fn,
        n_trials=args.n_trials,
        show_progress_bar=True,
        gc_after_trial=True
    )

    save_results(study, args.output)

    print_summary(study)

    # Save best params 
    if study.best_trial:
        best_params_path = args.output.replace('.json', '_best_params.json')
        with open(best_params_path, 'w') as f:
            json.dump({
                'trial_id': study.best_trial.number,
                'solve_rate': study.best_trial.value,
                'params': study.best_trial.params,
                'user_attrs': study.best_trial.user_attrs
            }, f, indent=2)
        print(f"Best params saved to {best_params_path}")


if __name__ == "__main__":
    main()
import argparse
import json
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from pathlib import Path
from datetime import datetime
from functools import partial

from sweep.objective import objective


def save_results(study: optuna.Study, output_path: str):
    """Save all trial results to JSON."""
    results = []

    for trial in study.trials:
        result = {
            'trial_id': trial.number,
            'state': trial.state.name,
            'value': trial.value,
            **trial.params,
            **trial.user_attrs,
        }
        results.append(result)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to {output_path}")


def print_summary(study: optuna.Study):
    """Print sweep summary statistics."""
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
            if isinstance(value, float):
                print(f"    {key}: {value:.6g}")
            else:
                print(f"    {key}: {value}")

        sorted_trials = sorted(completed, key=lambda t: t.value or 0, reverse=True)
        print(f"\nTop 5 trials:")
        for i, trial in enumerate(sorted_trials[:5], 1):
            print(f"  {i}. Trial #{trial.number}: solve_rate={trial.value:.4f}")


def main():
    parser = argparse.ArgumentParser(description="HRM Hyperparameter Sweep")
    parser.add_argument('--n_trials', type=int, default=60)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--train_json', type=str, default='data/StateTrainingSet.json')
    parser.add_argument('--state0_json', type=str, default='data/State0TrainingSet.json')
    parser.add_argument('--val_json', type=str, default='data/StateValSet.json')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--study_name', type=str, default='hrm_sweep')
    parser.add_argument('--prune', action='store_true', help='Enable median pruning')
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
    print(f"Pruning: {'enabled' if args.prune else 'disabled'}")
    print("=" * 60)

    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3) if args.prune else optuna.pruners.NopPruner()

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
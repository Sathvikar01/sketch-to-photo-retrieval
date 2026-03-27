"""
Quick ablation experiments for validation (reduced epochs for faster execution).
"""

import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

python_exe = sys.executable

EXPERIMENTS = [
    {
        'name': 'baseline',
        'description': 'Original training setup (10 epochs)',
        'args': {
            'epochs': 10,
            'batch_size': 16,
            'lr': 1e-4,
        }
    },
    {
        'name': 'augmented',
        'description': 'With modality-aware augmentations (10 epochs)',
        'args': {
            'epochs': 10,
            'batch_size': 16,
            'lr': 1e-4,
        }
    },
    {
        'name': 'full_quick',
        'description': 'All improvements - quick validation (20 epochs)',
        'args': {
            'epochs': 20,
            'batch_size': 32,
            'lr': 1e-4,
            'warmup_epochs': 3,
            'patience': 10,
        }
    },
]


def run_training(exp_name: str, args: dict) -> dict:
    """Run training for a single experiment."""
    import subprocess
    
    print(f"\n{'='*70}")
    print(f"Training: {exp_name}")
    print(f"{'='*70}")
    
    cmd = [python_exe, 'train.py', '--experiment_name', exp_name]
    for k, v in args.items():
        cmd.extend([f'--{k}', str(v)])
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    duration = time.time() - start
    
    if result.returncode != 0:
        print(f"Error: {result.stderr[:500]}")
        return {'status': 'failed', 'error': result.stderr[:500]}
    
    print(f"Completed in {duration/60:.1f} minutes")
    return {'status': 'success', 'duration': duration}


def run_eval(exp_name: str) -> dict:
    """Run evaluation on trained model."""
    from evaluation_metrics import evaluate_retrieval
    
    checkpoint = os.path.join('checkpoints', exp_name, 'best_model.pth')
    if not os.path.exists(checkpoint):
        checkpoint = os.path.join('checkpoints', exp_name, 'final_model.pth')
    
    if not os.path.exists(checkpoint):
        return {}
    
    print(f"\nEvaluating {exp_name}...")
    metrics = evaluate_retrieval(
        model_checkpoint=checkpoint,
        results_path=os.path.join('checkpoints', exp_name, 'eval.json'),
        verbose=True
    )
    return metrics


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'ablation_quick_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    print("="*70)
    print("QUICK ABLATION STUDY")
    print("="*70)
    
    all_results = []
    
    for exp in EXPERIMENTS:
        train_result = run_training(exp['name'], exp['args'])
        
        if train_result['status'] == 'success':
            metrics = run_eval(exp['name'])
            all_results.append({
                'name': exp['name'],
                'description': exp['description'],
                'metrics': metrics,
                'duration': train_result.get('duration', 0)
            })
        else:
            all_results.append({
                'name': exp['name'],
                'description': exp['description'],
                'error': train_result.get('error', 'Unknown'),
                'duration': train_result.get('duration', 0)
            })
    
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Experiment':<20} {'R@1':>10} {'R@5':>10} {'MRR':>10}")
    print("-"*70)
    for r in all_results:
        if 'metrics' in r and r['metrics']:
            m = r['metrics']
            print(f"{r['name']:<20} {m.get('recall@1', 0)*100:>9.2f}% {m.get('recall@5', 0)*100:>9.2f}% {m.get('mrr', 0):>10.4f}")
        else:
            print(f"{r['name']:<20} {'FAILED':>10}")
    print("-"*70)
    print(f"\nResults saved to: {results_dir}/")


if __name__ == '__main__':
    main()

"""
Run ablation experiments to validate each improvement component.

This script runs controlled experiments:
1. Baseline (current model)
2. + Augmentations only
3. + Hard negative mining only  
4. + Scheduler/optimization only
5. Full upgraded training stack
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime

EXPERIMENTS = [
    {
        'name': 'baseline',
        'description': 'Original training setup (random negatives, basic transforms, 10 epochs)',
        'args': {
            'epochs': 10,
            'batch_size': 16,
            'lr': 1e-4,
            'margin': 0.5,
        }
    },
    {
        'name': 'augmentations',
        'description': 'Enhanced modality-aware augmentations',
        'args': {
            'epochs': 10,
            'batch_size': 16,
            'lr': 1e-4,
            'margin': 0.5,
        }
    },
    {
        'name': 'hard_mining',
        'description': 'Batch-hard negative mining with combined loss',
        'args': {
            'epochs': 10,
            'batch_size': 32,
            'lr': 1e-4,
            'margin': 0.5,
        }
    },
    {
        'name': 'scheduler',
        'description': 'Cosine annealing LR with warmup',
        'args': {
            'epochs': 50,
            'batch_size': 32,
            'lr': 1e-4,
            'margin': 0.5,
        }
    },
    {
        'name': 'full_stack',
        'description': 'All improvements combined (aug + hard mining + scheduler + AMP)',
        'args': {
            'epochs': 50,
            'batch_size': 32,
            'lr': 1e-4,
            'margin': 0.5,
        }
    },
]


def run_experiment(exp_config: dict, seed: int = 42) -> dict:
    """
    Run a single experiment and return results.
    
    Args:
        exp_config: Experiment configuration dict
        seed: Random seed
    
    Returns:
        Dict with experiment results
    """
    exp_name = exp_config['name']
    print(f"\n{'='*70}")
    print(f"Running experiment: {exp_name}")
    print(f"Description: {exp_config['description']}")
    print(f"{'='*70}")
    
    import sys
    python_exe = sys.executable
    
    cmd = [
        python_exe, 'train.py',
        '--experiment_name', exp_name,
        '--seed', str(seed),
    ]
    
    for key, value in exp_config['args'].items():
        cmd.extend([f'--{key}', str(value)])
    
    if exp_name == 'full_stack':
        cmd.append('--use_amp')
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        
        if result.returncode != 0:
            print(f"Training failed with error:")
            print(result.stderr)
            return {
                'name': exp_name,
                'status': 'failed',
                'error': result.stderr,
                'duration': time.time() - start_time
            }
        
        history_path = os.path.join('checkpoints', exp_name, 'history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
        else:
            history = {}
        
        return {
            'name': exp_name,
            'status': 'success',
            'duration': time.time() - start_time,
            'history': history,
            'final_train_loss': history.get('train_loss', [None])[-1] if history.get('train_loss') else None,
            'best_val_recall@1': max(history.get('val_recall@1', [0])) if history.get('val_recall@1') else None,
        }
    
    except subprocess.TimeoutExpired:
        return {
            'name': exp_name,
            'status': 'timeout',
            'duration': time.time() - start_time
        }
    except Exception as e:
        return {
            'name': exp_name,
            'status': 'error',
            'error': str(e),
            'duration': time.time() - start_time
        }


def run_evaluation(exp_name: str) -> dict:
    """
    Run evaluation on trained model.
    
    Args:
        exp_name: Experiment name
    
    Returns:
        Evaluation metrics
    """
    import torch
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from evaluation_metrics import evaluate_retrieval, plot_cmc_curve, plot_rank_distribution
    
    checkpoint_path = os.path.join('checkpoints', exp_name, 'best_model.pth')
    
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join('checkpoints', exp_name, 'final_model.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found for {exp_name}")
        return {}
    
    print(f"\nEvaluating {exp_name}...")
    
    results_path = os.path.join('checkpoints', exp_name, 'evaluation_results.json')
    
    metrics = evaluate_retrieval(
        model_checkpoint=checkpoint_path,
        results_path=results_path,
        verbose=True
    )
    
    if metrics:
        cmc_path = os.path.join('checkpoints', exp_name, 'cmc_curve.png')
        plot_cmc_curve(metrics, cmc_path)
        
        with open(results_path, 'r') as f:
            results_data = json.load(f)
        rank_path = os.path.join('checkpoints', exp_name, 'rank_distribution.png')
        plot_rank_distribution(results_data['per_query_results'], rank_path)
    
    return metrics


def generate_ablation_table(results: list) -> str:
    """
    Generate LaTeX table for ablation results.
    
    Args:
        results: List of experiment results
    
    Returns:
        LaTeX table string
    """
    table = r"""
\begin{table}[h]
\centering
\caption{Ablation Study Results on CUFS Dataset}
\label{tab:ablation}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{Recall@1} & \textbf{Recall@5} & \textbf{MRR} \\
\midrule
"""
    
    for res in results:
        name = res['name'].replace('_', ' ').title()
        metrics = res.get('metrics', {})
        
        r1 = metrics.get('recall@1', 0)
        r5 = metrics.get('recall@5', 0)
        mrr = metrics.get('mrr', 0)
        
        if isinstance(r1, (int, float)):
            table += f"{name} & {r1:.4f} & {r5:.4f} & {mrr:.4f} \\\\\n"
        else:
            table += f"{name} & - & - & - \\\\\n"
    
    table += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return table


def main():
    """
    Run all ablation experiments and generate report.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'ablation_results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    print("="*70)
    print("ABLATION STUDY: Validating Model Improvements")
    print("="*70)
    print(f"Results will be saved to: {results_dir}")
    
    all_results = []
    
    for exp in EXPERIMENTS:
        exp_result = run_experiment(exp)
        
        if exp_result['status'] == 'success':
            metrics = run_evaluation(exp['name'])
            exp_result['metrics'] = metrics
        
        all_results.append(exp_result)
    
    summary = {
        'timestamp': timestamp,
        'experiments': all_results
    }
    
    with open(os.path.join(results_dir, 'ablation_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    latex_table = generate_ablation_table(all_results)
    with open(os.path.join(results_dir, 'ablation_table.tex'), 'w') as f:
        f.write(latex_table)
    
    print("\n" + "="*70)
    print("ABLATION STUDY COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {results_dir}/")
    print("\nSummary:")
    print("-"*70)
    
    for res in all_results:
        name = res['name']
        status = res['status']
        duration = res.get('duration', 0)
        
        if status == 'success' and 'metrics' in res:
            m = res['metrics']
            print(f"{name:20s}: R@1={m.get('recall@1', 0):.4f}, R@5={m.get('recall@5', 0):.4f}, MRR={m.get('mrr', 0):.4f} ({duration/60:.1f} min)")
        else:
            print(f"{name:20s}: {status} ({duration/60:.1f} min)")
    
    print("-"*70)
    
    return all_results


if __name__ == '__main__':
    main()

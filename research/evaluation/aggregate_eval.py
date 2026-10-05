"""Aggregate a verified LIBERO-10 task array into one 100-episode baseline."""
import json
import argparse
from pathlib import Path
from platforms.paths import output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("array_id")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    array_id = args.array_id
    assert array_id.isdecimal(), array_id
    root = output_path('smolvla', 'vanilla')
    summary = {
        'array_job_id': array_id,
        'suite': 'libero_10',
        'episodes_per_task': 10,
        'seed': 1000,
        'tasks': [],
    }
    for task_id in range(10):
        folder = root / f'libero10-task{task_id}-{array_id}_{task_id}'
        verification = json.loads((folder / 'verification.json').read_text())
        info = json.loads((folder / 'eval_info.json').read_text())
        assert len(info['per_task']) == 1, folder
        task_info = info['per_task'][0]
        assert task_info['task_group'] == 'libero_10', task_info
        assert task_info['task_id'] == task_id, task_info
        assert verification['n_episodes'] == 10, folder
        assert verification['verified_videos'] == 10, folder
        assert verification['n_success'] == task_info['n_success'], folder
        summary['tasks'].append({
            'task_id': task_id,
            'task_description': task_info['task_description'],
            'n_episodes': 10,
            'n_success': verification['n_success'],
            'pc_success': verification['pc_success'],
            'eval_s': verification['eval_s'],
            'verified_videos': verification['verified_videos'],
            'output_dir': str(folder),
        })
    summary['n_episodes'] = sum(t['n_episodes'] for t in summary['tasks'])
    summary['n_success'] = sum(t['n_success'] for t in summary['tasks'])
    summary['pc_success'] = 100 * summary['n_success'] / summary['n_episodes']
    summary['verified_videos'] = sum(t['verified_videos'] for t in summary['tasks'])
    assert summary['n_episodes'] == summary['verified_videos'] == 100
    out = args.output_dir or root / f'libero10-array-{array_id}'
    out.mkdir(parents=True, exist_ok=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

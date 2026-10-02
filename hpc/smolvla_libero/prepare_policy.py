"""Resolve pinned downloaded weights without network or changing model parameters."""
import json
from pathlib import Path

root = Path('/scratch/xzhan742/thesis')
paths = json.loads((root / 'manifests/model-paths.json').read_text())
baseline = json.loads(Path('hpc/smolvla_libero/local-baseline.json').read_text())
for key in ('policy', 'backbone'):
    assert Path(paths[key]).name == baseline[key]['revision'], (key, paths[key])
    assert Path(paths[key]).is_dir(), paths[key]
source = Path(paths['policy'])
target = root / 'models/smolvla-libero-local'
target.mkdir(parents=True, exist_ok=True)
for entry in source.iterdir():
    if entry.name in ('config.json', 'policy_preprocessor.json'):
        continue
    link = target / entry.name
    if not link.exists():
        link.symlink_to(entry)
    else:
        assert link.is_symlink() and link.resolve() == entry.resolve(), link
config = json.loads((source / 'config.json').read_text())
config['vlm_model_name'] = paths['backbone']
(target / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
preprocessor = json.loads((source / 'policy_preprocessor.json').read_text())
tokenizer_steps = [
    step for step in preprocessor['steps']
    if step['registry_name'] == 'tokenizer_processor'
]
assert len(tokenizer_steps) == 1, tokenizer_steps
assert tokenizer_steps[0]['config']['tokenizer_name'] == baseline['backbone']['repo']
tokenizer_steps[0]['config']['tokenizer_name'] = paths['backbone']
preprocessor_path = target / 'policy_preprocessor.json'
if preprocessor_path.is_symlink():
    preprocessor_path.unlink()
preprocessor_path.write_text(json.dumps(preprocessor, indent=2) + '\n')
print('Pinned policy:', paths['policy'])
print('Pinned backbone:', paths['backbone'])

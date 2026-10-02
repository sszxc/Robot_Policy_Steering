"""Validate completed rollout metrics and decode recorded videos."""
import json
import sys
from pathlib import Path

import av

output = Path(sys.argv[1])
expected = int(sys.argv[2])
metrics = json.loads((output / 'eval_info.json').read_text())
overall = metrics['overall']
assert overall['n_episodes'] == expected, overall
assert 0 <= overall['n_success'] <= expected, overall
videos = list(output.glob('videos/**/*.mp4'))
assert len(videos) == expected, (len(videos), expected)
for video in videos:
    assert video.stat().st_size > 0, video
    with av.open(str(video)) as container:
        frame = next(container.decode(video=0))
        assert frame.width > 0 and frame.height > 0, video
summary = {k: v for k, v in overall.items() if 'video' not in k}
summary['verified_videos'] = len(videos)
(output / 'verification.json').write_text(json.dumps(summary, indent=2) + '\n')
print('Verified evaluation:', json.dumps(summary, indent=2))

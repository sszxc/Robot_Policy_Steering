"""Check allocated CUDA execution, headless rendering and video dependencies."""
import json
import os
from pathlib import Path

import av
import mujoco
import numpy as np
import torch
import torchcodec
from OpenGL import GL

assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
x = torch.ones((64, 64), device='cuda')
assert float((x @ x)[0, 0]) == 64
torch.cuda.synchronize()
model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><light pos="0 0 3"/><geom type="sphere" size="0.1" '
    'rgba="1 0 0 1"/></worldbody></mujoco>'
)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
with mujoco.Renderer(model, height=64, width=64) as renderer:
    renderer.update_scene(data)
    frame = renderer.render()
    assert frame.shape == (64, 64, 3) and frame.dtype == np.uint8
    assert frame.max() > 0, 'Blank render'
    # MuJoCo may leave PyOpenGL without a current context after rendering.
    # A real nonblank frame with MUJOCO_GL=egl is the relevant render check.
    renderer_bytes = GL.glGetString(GL.GL_RENDERER)
    gl_renderer = renderer_bytes.decode() if renderer_bytes else 'unavailable'
info = {
    'job_id': os.environ['SLURM_JOB_ID'],
    'hostname': os.uname().nodename,
    'torch': torch.__version__,
    'cuda_runtime': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(),
    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    'egl_device_id': os.environ.get('MUJOCO_EGL_DEVICE_ID'),
    'gl_renderer': gl_renderer,
    'mujoco': mujoco.__version__,
    'torchcodec': torchcodec.__version__,
    'av': av.__version__,
}
print(json.dumps(info, indent=2))
(Path(os.environ['RPS_DATA_ROOT']) / 'manifests' / ('preflight-' + info['job_id'] + '.json')).write_text(
    json.dumps(info, indent=2) + '\n'
)

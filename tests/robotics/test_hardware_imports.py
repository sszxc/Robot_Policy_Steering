import subprocess
import sys


def test_hardware_modules_import_without_loading_device_interfaces():
    subprocess.run([sys.executable, "-c", """
import builtins
import importlib
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith('rtde_'):
        raise AssertionError('Hardware dependency loaded during import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
for name in ('ur5_read', 'ur5_move', 'ur5_move_servo'):
    importlib.import_module('robotics.hardware.ur5.' + name)
"""], check=True)

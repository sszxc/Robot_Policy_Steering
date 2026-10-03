import sys

from robotics.policies.scripted import spray


def test_spray_task_succeeds(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["spray"])
    assert spray.main()

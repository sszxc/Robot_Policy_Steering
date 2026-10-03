"""Runs the scripted alignment demos headless and checks their errors."""
from robotics.policies.scripted import align


def test_align_demos():
    res = align.main()
    assert abs(res["slide -1.0 cm"]) < 0.002 and abs(res["slide -2.0 cm"]) < 0.002
    assert abs(res["pivot +8 deg"]) < 1.5 and abs(res["pivot +16 deg"]) < 1.5
    assert all(v < 0.003 for k, v in res.items() if k.startswith("translate"))

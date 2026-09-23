from tasks import hammer


def test_hammer_drives_nail():
    depth, held = hammer.main()
    assert held and depth >= hammer.TARGET_DEPTH

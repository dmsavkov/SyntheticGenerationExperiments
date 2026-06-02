from routers.playground.ids import new_row_id


def test_new_row_id_format_and_unique():
    a = new_row_id()
    b = new_row_id()
    assert a.startswith("synth_")
    assert a != b
    assert len(a) == len("synth_") + 10

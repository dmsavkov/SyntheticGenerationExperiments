from routers.synthetic.batching import dynamic_batches, exemplar_char_count, prepare_exemplar


def test_dynamic_batch_respects_count():
    rows = [{"context": "a" * 50, "question": "", "options": "", "gold": "X"} for _ in range(25)]
    batches = dynamic_batches(rows)
    assert all(len(b) <= 10 for b in batches)
    assert sum(len(b) for b in batches) == 25


def test_exemplar_chars():
    row = prepare_exemplar({"context": "x" * 1000, "question": "", "options": "", "gold": "M"})
    assert row["exemplar_chars"] <= 600

from routers.playground.source_filter import row_matches_source


def test_arc_med_mmlu_preset():
    assert row_matches_source("ArcMMLU_30", "arc_med_mmlu")
    assert row_matches_source("MedMCQA_145", "arc_med_mmlu")
    assert row_matches_source("MMLU_management_86", "arc_med_mmlu")
    assert not row_matches_source("OpenTDB_General Knowledge_1", "arc_med_mmlu")


def test_opentdb_preset():
    assert row_matches_source("OpenTDB_Science_1", "opentdb_only")
    assert not row_matches_source("ArcMMLU_1", "opentdb_only")


def test_all_preset():
    assert row_matches_source("anything", "all")

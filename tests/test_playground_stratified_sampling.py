import pandas as pd

from routers.playground.sampling import stratified_by_dataset


def test_stratified_by_dataset_preserves_labels():
    df = pd.DataFrame(
        {
            "Global Index": [f"S_{i}" for i in range(12)],
            "Domain": ["A", "A", "A", "A", "B", "B", "B", "B", "A", "A", "B", "B"],
            "Dataset name": ["S1"] * 6 + ["S2"] * 6,
        }
    )
    ids = df["Global Index"].tolist()
    id_to_idx = {rid: i for i, rid in enumerate(ids)}
    chosen = stratified_by_dataset(df, ids, n=4, id_to_idx=id_to_idx, seed=42)
    assert len(chosen) == 4
    labels = {df.iloc[id_to_idx[i]]["Domain"] for i in chosen}
    assert len(labels) >= 1

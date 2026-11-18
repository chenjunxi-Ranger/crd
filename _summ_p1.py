import pandas as pd

df = pd.concat([pd.read_csv("p1_full_results.csv"),
                pd.read_csv("p1_seeds34_results.csv")])
print(df.shape, "seeds", sorted(df.seed.unique()))
for nz in sorted(df.noise.unique()):
    d = df[(df.noise == nz) & (df.budget == 0.5)]
    print(f"\nnoise={nz} budget=0.5")
    print(d.groupby("arm")[["val_acc", "val_ap", "val_auc",
                            "noise_removed", "clean_recall"]]
          .mean().round(4).sort_values("val_acc", ascending=False).to_string())

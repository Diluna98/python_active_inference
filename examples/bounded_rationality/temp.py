import pandas as pd

files = [
    "results_res_2.csv",
    "results_res_5.csv",
    "results_res_10.csv",
    "results_res_20.csv"
]

for file in files:
    df = pd.read_csv(file)

    filtered = df[df["step"] % 3 == 0]

    new_name = file.replace(".csv", "_filtered.csv")
    filtered.to_csv(new_name, index=False)
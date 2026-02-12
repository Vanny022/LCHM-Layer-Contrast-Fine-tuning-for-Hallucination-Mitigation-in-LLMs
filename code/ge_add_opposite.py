import pandas as pd

df = pd.read_csv("/home/weli0373/DoLa/general_splits_csv/val.csv")

df["oppositeanswer"] = df["hallucination"].apply(lambda x: "no" if str(x).strip().lower() == "yes" else "yes")

df.to_csv("/home/weli0373/DoLa/general_splits_csv/val.csv", index=False)

print("Done")

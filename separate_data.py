# save as: split_csv_simple.py
import os, argparse, random, math
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="输入CSV路径")
    ap.add_argument("--outdir", required=True, help="输出目录")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratio", type=str, default="8,1,1", help="比例，默认8,1,1")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input)
    n = len(df)

    # 随机打乱
    random.seed(args.seed)
    idx = list(range(n))
    random.shuffle(idx)

    r = [int(x) for x in args.ratio.split(",")]
    s = sum(r)
    n_train = int(round(n * r[0] / s))
    n_val   = int(round(n * r[1] / s))
    n_test  = n - n_train - n_val

    idx_train = idx[:n_train]
    idx_val   = idx[n_train:n_train+n_val]
    idx_test  = idx[n_train+n_val:]

    df.iloc[idx_train].to_csv(os.path.join(args.outdir, "train.csv"), index=False)
    df.iloc[idx_val].to_csv(os.path.join(args.outdir, "val.csv"), index=False)
    df.iloc[idx_test].to_csv(os.path.join(args.outdir, "test.csv"), index=False)

    print(f"train: {len(idx_train)}, val: {len(idx_val)}, test: {len(idx_test)}")

if __name__ == "__main__":
    main()

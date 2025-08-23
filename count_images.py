import pandas as pd

FILESTEMS = open("bias_classes.txt").read().splitlines()

def count_gen_searched(df):
    df = df.drop_duplicates(subset="name", keep="first")
    num_syn = df.loc[df["imgpath"].str.contains("ai_images")].shape[0]
    return (num_syn, df.shape[0] - num_syn)

if __name__ == "__main__":
    results = pd.DataFrame(columns=["bias_class", "num_syn", "num_real", "num_total"])

    for bias_class in FILESTEMS:
        syn, real = count_gen_searched(pd.read_csv(f"./dictionary/{bias_class}_entity.csv"))
        results = results._append({
            "bias_class": bias_class,
            "num_syn": syn,
            "num_real": real,
            "num_total": syn + real
        }, ignore_index=True)

    total_syn = results["num_syn"].sum()
    total_real = results["num_real"].sum()
    results = results._append({
        "bias_class": "all",
        "num_syn": total_syn,
        "num_real": total_real,
        "num_total": total_syn + total_real
    }, ignore_index=True)

    results.to_csv("bias_class_counts.csv", index=False)
import pandas as pd
from pathlib import Path
from PIL import Image

BIAS_CLASSES = open("./bias_classes.txt").read().splitlines()

def generate_subset(ent_df):
    ent_df = ent_df.drop_duplicates(subset=['imgpath'], keep="first")
    for idx, row in ent_df.iterrows():
        if not Path(row['imgpath']).exists():
            print(f"Image path does not exist: {row['imgpath']}")
        # copy image at row["imgpath"] to ./subset/<row["name"]>.jpg
        img = Image.open(row['imgpath'])
        img.save(f"./subset/{row['name']}.jpg")

if __name__ == "__main__":
    for bc in BIAS_CLASSES:
        generate_subset(pd.read_csv(f"./dictionary/{bc}_entity.csv"))
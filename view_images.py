import pandas as pd
import os
import matplotlib.pyplot as plt

#FILESTEMS = open("./bias_classes.txt").read().splitlines()
FILESTEMS = ["Race_ethnicity"]

for filestem in FILESTEMS:
    df = pd.read_csv(f"./dictionary/{filestem}_entity.csv").drop_duplicates(subset=["name"], keep="first")
    print(df.shape)
    for idx, row in df.iterrows():
        name = row["name"]
        image_path = row["imgpath"]
        tags = row["tags"]
        if not os.path.exists(image_path):
            print(f"❌ Missing: {image_path}")
            continue
        
        # use matplotlib to show the image, and show tags
        img = plt.imread(image_path)
        plt.imshow(img)
        plt.title(name)
        # move tags to the bottom of the image
        plt.text(0.5, 0.80, tags, ha='center', va='bottom', fontsize=10)
        # show whether image is ai or searched
        if "ai" in image_path:
            plt.text(0.5, 0.95, "AI", ha='center', va='top', fontsize=10)
        else:
            plt.text(0.5, 0.95, "Searched", ha='center', va='top', fontsize=10)
        plt.axis('off')
        plt.show()
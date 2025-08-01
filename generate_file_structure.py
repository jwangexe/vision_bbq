import pandas as pd
from utils import is_probable_name, get_entity
import os

# ========== GLOBAL VARIABLES ==========
FILE_WEIGHT = 1_000_000
DATASETS_PATH = "./data/"
DICT_PATH = "./dictionary/"
VOID_CHOICES = [
    "unknown",
    "Can't answer",
    "Cannot answer",
    "Unknown",
    "Not known",
    "Not enough info",
    "Can't be determined",
    "Not answerable",
    "Not enough information",
    "Cannot be determined",
    "Undetermined"
]
BIAS_CLASSES = open("bias_classes.txt", "r").read().split("\n")

# ========== Data Processing ==========
def get_entity_dataframe(df, fileid):
    dat = []
    for ind, row in df.iterrows():
        bbq_id = (FILE_WEIGHT*fileid+row["example_id"])*10
        for i in range(3):
            colname = "ans"+str(i)
            entname = get_entity(row[colname], row["answer_info"][colname][1])
            if entname not in VOID_CHOICES:
                dat.append((bbq_id+i, entname, "", ""))
    out_df = pd.DataFrame.from_records(dat, columns=["bbq_id", "name", "tags", "imgpath"]).astype({"name": str, "bbq_id": int, "tags": str, "imgpath": str})
    return out_df


def generate_whole_dataset(fileid, filepath, savepath):
    df = pd.read_json(filepath, lines=True)
    get_entity_dataframe(df, fileid).to_csv(savepath)


def generate_sample_dataset(fileid, filepath, savepath, sample_size=50):
    df = pd.read_json(filepath, lines=True).sample(n=sample_size, random_state=42, axis=0, ignore_index=True)
    get_entity_dataframe(df, fileid).to_csv(savepath)


# ========== Execution ==========
if __name__ == "__main__":
    if input("Are you sure you want to generate new files(y/n)? ").lower()[0] == 'y':
        for fileid, filestem in enumerate(BIAS_CLASSES):
            srcpath = os.path.join(DATASETS_PATH, filestem+".jsonl")
            dstpath = os.path.join(DICT_PATH, filestem+"_entity.csv")
            classespath = os.path.join(DICT_PATH, filestem+"_classes.csv")

            user_choice = input(f"Do you want to generate file {filestem}?\nPress 0 to not generate, 1 for full set, 2 for sample:")
            if(user_choice[0] == "1"): generate_whole_dataset(fileid, srcpath, dstpath)
            elif(user_choice[0] == "2"): generate_sample_dataset(fileid, srcpath, dstpath)
            else: continue
    
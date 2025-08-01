import pandas as pd
from pathlib import Path
from openai import OpenAI
from utils import get_entity, encode_image_to_base64

# Load API key and initialize client
APIKEY_PATH = "./api_keys/openai/api_key.txt"
api_key = open(APIKEY_PATH).read().strip()
client = OpenAI(api_key=api_key)

FILSTEMS = [
    "Age", "Disability_status", "Gender_identity", "Nationality", "Physical_appearance",
    "Race_ethnicity", "Race_x_gender", "Race_x_SES", "Religion", "SES", "Sexual_orientation"
]

LABEL_MAP = {'A': 0, 'B': 1, 'C': 2}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def get_model_answer(context, question, answer_texts, model_name, path1, path2, with_images):
    schema = {
        "name": "choose_option",
        "description": "Choose the letter (A, B, or C) of the best answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "option": {
                    "type": "string",
                    "enum": ["A", "B", "C"]
                }
            },
            "required": ["option"]
        }
    }

    choices = "\n".join(f"{l}. {a}" for l, a in zip(['A', 'B', 'C'], answer_texts))
    prompt = f"Context: {context}\nQuestion: {question}\nChoices:\n{choices}"

    try:
        user_content = [ {"type": "text", "text": prompt} ]
        if with_images:
            user_content += [
                {"type": "image_url", "image_url": {"url": encode_image_to_base64(path1)}},
                {"type": "image_url", "image_url": {"url": encode_image_to_base64(path2)}}
            ]

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": user_content}],
            tools=[{"type": "function", "function": schema}],
            tool_choice="auto"
        )

        letter = eval(response.choices[0].message.tool_calls[0].function.arguments)["option"]
        return LABEL_MAP.get(letter, -1)
    except Exception as e:
        print(f"API error: {e}")
        return -1

def get_prediction_df(df, entities_df, save_csv_path, model_name, with_images):
    gpt_labels = []

    for i in range(3):
        if f"orig_ans{i}" not in df.columns:
            df[f"orig_ans{i}"] = df[f"ans{i}"]

    for idx, row in df.iterrows():
        context, question, label = row['context'], row['question'], row['label']
        if isinstance(label, str):
            label = LABEL_MAP.get(label.upper(), -1)
        answer_info = row['answer_info']
        if type(answer_info) == str:
            answer_info = eval(answer_info)
        answer_texts = [row[f'ans{i}'] for i in range(3)]
        answer_entities = []
        for i in range(3):
            colname = f"ans{i}"
            if answer_info[colname][1] == "unknown":
                continue
            answer_entities.append(get_entity(row["orig_"+colname], answer_info[colname][1]))
        
        #print(answer_entities, entities_df.loc[entities_df["name"] == answer_entities[0]], entities_df.loc[entities_df["name"] == answer_entities[1]])
        try:
            path1 = entities_df.loc[entities_df["name"] == answer_entities[0]].iloc[0]["imgpath"]
            path2 = entities_df.loc[entities_df["name"] == answer_entities[1]].iloc[0]["imgpath"]
        except IndexError:
            # label of -1 means that it was not evaluated
            gpt_labels.append(-1)
            continue

        model_label = get_model_answer(context, question, answer_texts, model_name, path1, path2, with_images)
        gpt_labels.append(model_label)

        pred_chr = REVERSE_LABEL_MAP.get(model_label, "?")
        true_chr = REVERSE_LABEL_MAP.get(label, "?")
        is_correct = model_label == label
        print(f"[{idx+1}] → predicted: {pred_chr}, true: {true_chr}, {'✓' if is_correct else '✗'}")

    df["gpt_label"] = gpt_labels
    Path(save_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_csv_path, index=False)    
    
    return df


def process_filestem(filestem, model_name, sample_size):

    obf_df = pd.read_csv(f"./data/{filestem}_prompts.csv")
    unobf_df = pd.read_json(f"./data/{filestem}.jsonl", lines=True)
    ent_df = pd.read_csv(f"./dictionary/{filestem}_entity.csv")

    if sample_size != -1:
        obf_df = obf_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        unobf_df = unobf_df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    print(f"---------- {filestem} ----------")
    # without images
    get_prediction_df(obf_df, ent_df, f"./results/{filestem}C.csv", model_name, False) # debiased
    get_prediction_df(unobf_df, ent_df, f"./results/{filestem}A.csv", model_name, False) # not debiased
    # with images
    get_prediction_df(obf_df, ent_df, f"./results/{filestem}B.csv", model_name, True) # debiased
    get_prediction_df(unobf_df, ent_df, f"./results/{filestem}D.csv", model_name, True) # not debiased




if __name__ == "__main__":
    all_summaries = []
    for filestem in FILSTEMS:
        if input(f"Process {filestem}? (y/n) ").lower().startswith("y"):
            try:
                quantity = int(input("How many samples should be taken? (-1 for all): "))
            except ValueError:
                print("Invalid number. Skipping.")
                continue
            all_summaries.append(process_filestem(filestem, "gpt-4o-mini", quantity))

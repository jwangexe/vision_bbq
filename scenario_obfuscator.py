import json
import pandas as pd
import re
from pathlib import Path

FILSTEMS = [
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Race_x_gender",
    "Race_x_SES",
    "Religion",
    "SES",
    "Sexual_orientation"
]

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

def get_name_order(template):
    order = []
    for match in re.finditer(r'\{\{NAME(\d)\}\}', template):
        order.append((f'NAME{match.group(1)}', match.start()))
    return [x[0] for x in sorted(order, key=lambda y: y[1])]

def replace_groups_with_images(text, name1, name2):
    def replace_name(text, name, replacement):
        pattern = re.compile(rf'(?i)(?<!\w){re.escape(name)}(?!\w)')
        return pattern.sub(replacement, text)

    text = replace_name(text, name1, 'Image1')
    text = replace_name(text, name2, 'Image2')
    return text

def process_filestem(filestem):
    print(f"Processing filestem: {filestem}")

    template_path = f'template/new_templates - {filestem}.csv'
    input_jsonl_path = f'data/{filestem}.jsonl'
    output_jsonl_path = f'data/{filestem}_prompts.jsonl'
    output_csv_path = f'data/{filestem}_prompts.csv'

    templates = pd.read_csv(template_path)

    with open(input_jsonl_path, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]

    template_map = {}
    for _, row in templates.iterrows():
        qid = str(row['Q_id']).strip()
        template_map[qid] = {
            'ambig': row['Ambiguous_Context'],
            'disambig': row['Disambiguating_Context'] if 'Disambiguating_Context' in row and pd.notna(row['Disambiguating_Context']) else None
        }

    new_data = []
    for entry in data:
        qid = str(entry['question_index']).strip()
        if qid not in template_map:
            new_data.append(entry)
            continue

        template_context = template_map[qid]['ambig']
        template_disambig = template_map[qid]['disambig']

        name_order = get_name_order(template_context)

        # Save original answers before replacement
        for i in range(3):
            orig_key = f'orig_ans{i}'
            entry[orig_key] = entry.get(f'ans{i}', None)

        # Gather all answers and their texts
        answers = []
        for i in range(3):
            ans_text = entry['answer_info'].get(f'ans{i}', [None])[0]
            answers.append(ans_text)

        # Filter out VOID_CHOICES answers (case-insensitive)
        non_void_answers = [ans for ans in answers if ans and ans.lower() not in [v.lower() for v in VOID_CHOICES]]

        if len(non_void_answers) < 2:
            # Fallback to ans0 and ans2 if not enough non-void answers
            name_candidates = [entry['answer_info']['ans0'][0], entry['answer_info']['ans2'][0]]
        else:
            name_candidates = non_void_answers[:2]

        # Map to name1 and name2 according to name order in template
        if name_order == ['NAME1', 'NAME2']:
            name1, name2 = name_candidates
        else:
            name2, name1 = name_candidates

        entry['context'] = replace_groups_with_images(entry['context'], name1, name2)

        if 'disambiguating context' in entry:
            entry['disambiguating context'] = replace_groups_with_images(entry['disambiguating context'], name1, name2)
        elif 'disambiguating_context' in entry:
            entry['disambiguating_context'] = replace_groups_with_images(entry['disambiguating_context'], name1, name2)

        # Replace in answers if they are strings
        for k in ['ans0', 'ans1', 'ans2']:
            val = entry.get(k, None)
            if val and isinstance(val, str):
                if name1 in val:
                    entry[k] = val.replace(name1, 'Image1')
                if name2 in val:
                    entry[k] = entry[k].replace(name2, 'Image2')

        new_data.append(entry)

    Path(output_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl_path, 'w') as f_out:
        for row in new_data:
            f_out.write(json.dumps(row) + '\n')

    df = pd.DataFrame(new_data)
    df.to_csv(output_csv_path, index=False)

    print(f"Finished processing {filestem}. Output saved to:")
    print(f" - JSONL: {output_jsonl_path}")
    print(f" - CSV: {output_csv_path}\n")

def main():
    for filestem in FILSTEMS:
        if input(f"Would you like to process {filestem}? (y/n) ").lower().startswith('y'):
            process_filestem(filestem)

if __name__ == "__main__":
    main()

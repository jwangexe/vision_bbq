import ast

def safe_parse_gpt_list(gpt_output):
    # Remove leading/trailing whitespace
    gpt_output = gpt_output.strip()
    # If not starting with [ or ending with ], add them
    if not gpt_output.startswith("["):
        gpt_output = "[" + gpt_output
    if not gpt_output.endswith("]"):
        gpt_output = gpt_output + "]"
    try:
        # Safely evaluate the string as a Python list
        result = ast.literal_eval(gpt_output)
        # Ensure it's a list
        if isinstance(result, list):
            return result
        else:
            return []
    except Exception as e:
        print(f"Error parsing GPT output: {e}")
        return []


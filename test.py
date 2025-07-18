import re
import pandas as pd

_name_vocab_df = pd.read_csv('./data/vocabulary_proper_names.csv')
FIRST_NAMES = set(_name_vocab_df[_name_vocab_df['First_last'].str.contains('first', case=False, na=False)]['Name'].str.lower())
LAST_NAMES = set(_name_vocab_df[_name_vocab_df['First_last'].str.contains('last', case=False, na=False)]['Name'].str.lower())

def is_probable_name(s):
    words = s.strip().split()
    # Check if any words are in the name vocab (first or last)
    name_like = any(w.lower() in FIRST_NAMES or w.lower() in LAST_NAMES for w in words if w.isalpha())
    if name_like:
        return True
    return False

print(is_probable_name("The family of Ali Al-Mutairi, who's a receptionist"))
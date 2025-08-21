
from symspellpy import SymSpell, Verbosity
from symspellpy import SymSpell, Verbosity

from symspellpy import SymSpell, Verbosity
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = "new_name_dict.txt"
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

def is_startwith_upper(word):
    """Check if the word starts with an uppercase letter"""
    return word[0].isupper() if word else False
def check_and_correct_word(word):
    
    syllables = word.split(" ")
    for i, syllable in enumerate(syllables):
        if not is_startwith_upper(syllable):
            return word, False, "lowercase"
    word = word.strip().replace(" ", "_").lower()
    
    suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
    if suggestions:
        if word != suggestions[0].term:
            return word, False, suggestions[0].term
        else:
            return word, True, suggestions[0].term

    return word, True, "not_found"  # Return the original word if no suggestions found

sym_one_word = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
one_word_dict = "20250820_124744_vi_dict.txt"
sym_one_word.load_dictionary(one_word_dict, term_index=0, count_index=1)

def spell_check_one_word(word):
    """
    Check and correct a single word using the loaded dictionary.
    
    Args:
        word (str): The word to check and correct.
        
    Returns:
        tuple: A tuple containing the original word, a boolean indicating if it is correct,
               and the corrected word or suggestion.
    """
    word = word.strip().lower()
    
    suggestions = sym_one_word.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
    if suggestions:
        if word != suggestions[0].term:
            return word, False, suggestions[0].term
        else:
            return word, True, suggestions[0].term

    return word, True, "not_found"  # Return the original word if no suggestions found  
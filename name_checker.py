from symspellpy import SymSpell, Verbosity
from symspellpy import SymSpell, Verbosity

from symspellpy import SymSpell, Verbosity
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
def check_and_correct_word(word):
    
    suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
    if suggestions:
        if word != suggestions[0].term:
            return word, False, suggestions[0].term
        else:
            return word, True, suggestions[0].term

    return word, False, "not_found"  # Return the original word if no suggestions found
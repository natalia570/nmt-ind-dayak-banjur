import json
import random
from typing import Dict, List


def load_dictionary(path: str) -> Dict:
    """
    Load synonym dictionary from a JSON file.

    Expected JSON format:
    {
        "kata": {
            "sinonim": ["padanan1", "padanan2", ...]
        }
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_synonym(word: str, dictionary: Dict) -> str | None:
    """
    Get the first synonym of a word from the dictionary.
    """
    if word in dictionary and dictionary[word].get("sinonim"):
        synonyms = dictionary[word]["sinonim"]
        if len(synonyms) > 0:
            return synonyms[0]
    return None


def synonym_replacement(
    sentence: str,
    dictionary: Dict,
    num_replacements: int = 1
) -> str:
    """
    Apply synonym replacement to a sentence.

    Args:
        sentence: input sentence
        dictionary: synonym dictionary
        num_replacements: number of words to replace

    Returns:
        Augmented sentence
    """
    words: List[str] = sentence.split()
    new_sentence = words.copy()

    candidates = [
        i for i, word in enumerate(words)
        if get_synonym(word, dictionary)
    ]

    if not candidates:
        return sentence

    chosen_indices = random.sample(
        candidates,
        min(num_replacements, len(candidates))
    )

    for idx in chosen_indices:
        synonym = get_synonym(words[idx], dictionary)
        if synonym:
            new_sentence[idx] = synonym

    return " ".join(new_sentence)

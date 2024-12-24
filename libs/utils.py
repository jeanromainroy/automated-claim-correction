# Libraries
import json
import hashlib

# Saving the dictionary to a file, supports tuple keys
def save_dict_as_json(d, filepath, indent=4):
    converted_dict = {str(key): value for key, value in d.items()}
    with open(filepath, 'w') as file:
        json.dump(converted_dict, file, ensure_ascii=False, indent=indent)


# Loading the dictionary from a file, supports tuple keys
def load_dict_from_json(filepath):
    with open(filepath, 'r') as file:
        converted_dict = json.load(file)
    # Convert keys back to tuples
    return {eval(key): value for key, value in converted_dict.items()}


# Generate a consistent hash from a string
def consistent_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()

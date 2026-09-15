import json

# Load JSON file
with open("cal.json", "r") as file:
    data = json.load(file)

def count_key_value_pairs(obj):
    count = 0

    if isinstance(obj, dict):
        count += len(obj)  # Count current dictionary key-value pairs

        for value in obj.values():
            count += count_key_value_pairs(value)

    elif isinstance(obj, list):
        for item in obj:
            count += count_key_value_pairs(item)

    return count

total_pairs = count_key_value_pairs(data)

print(f"Total key-value pairs: {total_pairs}")
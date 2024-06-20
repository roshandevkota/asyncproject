def convert_lists_to_sets(obj):
    if isinstance(obj, list):
        return set(obj)
    if isinstance(obj, dict):
        return {key: convert_lists_to_sets(value) for key, value in obj.items()}
    if isinstance(obj, set):
        return {convert_lists_to_sets(item) for item in obj}
    return obj

def convert_sets_to_lists(obj):
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, dict):
        return {key: convert_sets_to_lists(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_sets_to_lists(item) for item in obj]
    return obj

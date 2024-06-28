def convert_lists_to_sets(obj):
    """
    Recursively converts lists to sets within a nested structure.

    Args:
        obj: The object to convert.

    Returns:
        The converted object with lists turned into sets.

    Example:
        >>> nested_list = {'a': [1, 2, 3], 'b': {'c': [4, 5, 6]}}
        >>> result = convert_lists_to_sets(nested_list)
        >>> print(result)
        {'a': {1, 2, 3}, 'b': {'c': {4, 5, 6}}}
        
    Why:
        This function is useful for converting lists to sets within complex
        nested structures, making them more suitable for certain operations.
    """
    if isinstance(obj, list):
        return set(obj)
    if isinstance(obj, dict):
        return {key: convert_lists_to_sets(value) for key, value in obj.items()}
    if isinstance(obj, set):
        return {convert_lists_to_sets(item) for item in obj}
    return obj

def convert_sets_to_lists(obj):
    """
    Recursively converts sets to lists within a nested structure.

    Args:
        obj: The object to convert.

    Returns:
        The converted object with sets turned into lists.

    Example:
        >>> nested_set = {'a': {1, 2, 3}, 'b': {'c': {4, 5, 6}}}
        >>> result = convert_sets_to_lists(nested_set)
        >>> print(result)
        {'a': [1, 2, 3], 'b': {'c': [4, 5, 6]}}
        
    Why:
        This function is useful for converting sets to lists within complex
        nested structures, making them more suitable for serialization.
    """
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, dict):
        return {key: convert_sets_to_lists(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_sets_to_lists(item) for item in obj]
    return obj

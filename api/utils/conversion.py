def convert_lists_to_sets(obj):
    """
    Recursively converts all lists within the given object to sets.

    This function takes an object, which can be a list, set, dictionary, or any other type, 
    and recursively converts all lists within that object to sets. It handles nested structures 
    such as dictionaries containing lists or sets, and lists or sets containing other lists or sets.

    Parameters
    ----------
    obj : any
        The object to be converted. This can be a list, set, dictionary, or any other type.

    Returns
    -------
    any
        The converted object with all lists turned into sets.

    Why
    ---
        These functions are used to handle the conversion between lists and 
        sets within a complex nested structure, such as metadata saved into a 
        JSON file. JSON does not support sets, so `convert_sets_to_lists` 
        ensures all sets are converted to lists before saving. 
        `convert_lists_to_sets` is then used to convert the lists back to sets 
        when loading the JSON, preserving the original data structure and 
        types.

    Examples
    --------
    >>> convert_lists_to_sets([1, 2, 3, 3])
    {1, 2, 3}

    >>> convert_lists_to_sets({'a': [1, 2], 'b': [2, 3, 3]})
    {'a': {1, 2}, 'b': {2, 3}}

    >>> convert_lists_to_sets({'a': {'b': [1, 2, 2]}, 'c': [3, 4]})
    {'a': {'b': {1, 2}}, 'c': {3, 4}}
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
    Recursively converts all sets within the given object to lists.

    This function takes an object, which can be a list, set, dictionary, or any other type, 
    and recursively converts all sets within that object to lists. It handles nested structures 
    such as dictionaries containing sets or lists, and sets or lists containing other sets or lists.

    Parameters
    ----------
    obj : any
        The object to be converted. This can be a list, set, dictionary, or any other type.

    Returns
    -------
    any
        The converted object with all sets turned into lists.

    Why
    ---
        These functions are used to handle the conversion between lists and 
        sets within a complex nested structure, such as metadata saved into a 
        JSON file. JSON does not support sets, so `convert_sets_to_lists` 
        ensures all sets are converted to lists before saving. 
        `convert_lists_to_sets` is then used to convert the lists back to sets 
        when loading the JSON, preserving the original data structure and 
        types.

    Examples
    --------
    >>> convert_sets_to_lists({1, 2, 3})
    [1, 2, 3]

    >>> convert_sets_to_lists({'a': {1, 2}, 'b': {2, 3}})
    {'a': [1, 2], 'b': [2, 3]}

    >>> convert_sets_to_lists({'a': {'b': {1, 2}}, 'c': {3, 4}})
    {'a': {'b': [1, 2]}, 'c': [3, 4]}
    """
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, dict):
        return {key: convert_sets_to_lists(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_sets_to_lists(item) for item in obj]
    return obj

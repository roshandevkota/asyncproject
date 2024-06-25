import pandas as pd

def read_excel_file(file_path):
    """
    Reads an Excel file and returns its content as a pandas DataFrame.

    This function reads an Excel file and checks if it contains a single sheet. 
    If the file has more than one sheet, it prints an error message and returns None. 
    If the file has a single sheet, it reads the sheet into a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        The path to the Excel file to be read.

    Returns
    -------
    pandas.DataFrame or None
        A DataFrame containing the data from the Excel sheet, or None if there is an error 
        or if the file contains multiple sheets.

    Why
    ---
    This function ensures that only single-sheet Excel files are processed, 
    simplifying the data loading process for applications that do not require multi-sheet 
    support. It also provides error handling to manage file reading issues gracefully.

    Examples
    --------
    >>> df = read_excel_file("data.xlsx")
    >>> if df is not None:
    ...     print(df.head())
    Error: The Excel file 'data.xlsx' contains multiple sheets. Only single-sheet files are supported.
    """

    try:
        xls = pd.ExcelFile(file_path)
        if len(xls.sheet_names) > 1:
            print(f"Error: The Excel file '{file_path}' contains multiple sheets. Only single-sheet files are supported.")
            return None
        else:
            sheet_name = xls.sheet_names[0]
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            return df
    except Exception as e:
        print(f"Failed to read Excel file {file_path}: {e}")
        return None




def read_text_or_csv(file_path, delimiter=','):
    """
    Reads a text or CSV file with various potential delimiters and returns its content as a pandas DataFrame.

    This function attempts to read a text or CSV file using a list of potential delimiters 
    (comma, tab, semicolon, space). If it successfully reads the file with any of these delimiters, 
    it returns a DataFrame. Otherwise, it prints an error message and returns None.

    Parameters
    ----------
    file_path : str
        The path to the text or CSV file to be read.
    delimiter : str, optional
        The initial delimiter to try. Default is comma (,).

    Returns
    -------
    pandas.DataFrame or None
        A DataFrame containing the data from the file, or None if there is an error 
        or if no valid delimiter is found.

    Why
    ---
    This function provides flexibility in reading text or CSV files with various delimiters, 
    which is useful when dealing with data from different sources where the delimiter is not known in advance.

    Examples
    --------
    >>> df = read_text_or_csv("data.csv")
    Successfully read text/CSV file data.csv with delimiter ','.
    >>> if df is not None:
    ...     print(df.head())
    """

    potential_delimiters = [delimiter, '\t', ';', ' ']
    for delim in potential_delimiters:
        try:
            df = pd.read_csv(file_path, sep=delim)
            print(f"Successfully read text/CSV file {file_path} with delimiter '{delim}'.")
            return df
        except pd.errors.ParserError:
            continue
    print(f"Failed to read text/CSV file {file_path} with any of the potential delimiters.")
    return None




def load_data_dynamic(file_path, delimiter=None):
    """
    Dynamically loads data from an Excel, CSV, or text file into a pandas DataFrame.

    This function determines the file type based on its extension and 
    uses the appropriate function to read the file. It supports Excel files (both .xlsx and .xls) 
    and text/CSV files. For text/CSV files, it can optionally detect the delimiter.

    Parameters
    ----------
    file_path : str
        The path to the file to be read.
    delimiter : str, optional
        The delimiter to use for text/CSV files. If None, the delimiter will be detected automatically.

    Returns
    -------
    pandas.DataFrame or None
        A DataFrame containing the data from the file, or None if there is an error 
        or if the file type is unsupported.

    Why
    ---
    This function provides a unified interface for loading data from different file formats, 
    simplifying the data ingestion process in applications that need to handle multiple file types.

    Examples
    --------
    >>> df = load_data_dynamic("data.xlsx")
    >>> if df is not None:
    ...     print(df.head())
    
    >>> df = load_data_dynamic("data.csv")
    Successfully read text/CSV file data.csv with delimiter ','.
    >>> if df is not None:
    ...     print(df.head())
    """

    extension = file_path.split('.')[-1].lower()
    if extension in ['xlsx', 'xls']:
        return read_excel_file(file_path)
    elif extension in ['csv', 'txt']:
        if delimiter is None:
            delimiter = detect_separator(file_path)
        return read_text_or_csv(file_path, delimiter)
    else:
        print("Unsupported file type")
        return None




def detect_separator(file_path):
    """
    Detects the delimiter used in a text or CSV file.

    This function reads the first few lines of a file and checks for consistency 
    of potential delimiters (comma, tab, semicolon, space). It returns the detected 
    delimiter if one is found to be consistent.

    Parameters
    ----------
    file_path : str
        The path to the file whose delimiter needs to be detected.

    Returns
    -------
    str or None
        The detected delimiter, or None if no consistent delimiter is found.

    Why
    ---
    Automatically detecting the delimiter in a text or CSV file is useful 
    for handling files from different sources without needing prior knowledge 
    of their structure.

    Examples
    --------
    >>> detect_separator("data.csv")
    ','
    """

    potential_delimiters = [',', '\t', ';', ' ']
    sample_size = 10

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            lines = [line for line, _ in zip(file, range(sample_size))]
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

    for delim in potential_delimiters:
        first_line_columns = len(lines[0].split(delim))
        delimiter_is_consistent = all(len(line.split(delim)) == first_line_columns for line in lines)
        
        if delimiter_is_consistent and first_line_columns > 1:
            return delim

    print(f"No consistent delimiter found in {file_path}.")
    return None





def detect_separator_content(file_content):
    """
    Detects the delimiter used in a string containing file content.

    This function analyzes the first few lines of a string and checks for consistency 
    of potential delimiters (comma, tab, semicolon, space). It returns the detected 
    delimiter if one is found to be consistent.

    Parameters
    ----------
    file_content : str
        The content of the file as a string.

    Returns
    -------
    str or None
        The detected delimiter, or None if no consistent delimiter is found.

    Why
    ---
    Automatically detecting the delimiter in a string containing file content is useful 
    for processing data in memory, such as when files are uploaded through a web interface.

    Examples
    --------
    >>> file_content = "col1,col2,col3\\n1,2,3\\n4,5,6"
    >>> detect_separator_content(file_content)
    ','
    """
    
    potential_delimiters = [',', '\t', ';', ' ']
    sample_size = 10

    lines = file_content.splitlines()[:sample_size]

    for delim in potential_delimiters:
        first_line_columns = len(lines[0].split(delim))
        delimiter_is_consistent = all(len(line.split(delim)) == first_line_columns for line in lines)
        
        if delimiter_is_consistent and first_line_columns > 1:
            return delim

    print(f"No consistent delimiter found.")
    return None
import pandas as pd

def read_excel_file(file_path):
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

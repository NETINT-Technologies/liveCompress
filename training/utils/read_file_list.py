"""
Reads a file list of folders and returns a list of folders.
"""
def read_file_list(file_path):
    with open(file_path, 'r') as file:
        folders = [line.strip() for line in file.readlines()]
    return folders
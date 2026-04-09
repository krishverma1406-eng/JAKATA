import os
from typing import Any

TOOL_DEFINITION = {
    "name": "folder_stats",
    "description": "Reports the number of files and the total size of a specified directory.",
    "parameters": {
        "directory_path": {
            "type": "string",
            "description": "The path of the directory to analyze."
        }
    }
}

def execute(params: dict[str, Any]) -> dict[str, Any]:
    """
    Analyze the specified directory and return the number of files and total size.

    Args:
        params (dict): A dictionary containing 'directory_path' as a key.

    Returns:
        dict: A dictionary with 'file_count' and 'total_size' in bytes.
    """
    directory_path = params.get("directory_path")
    
    if not directory_path or not os.path.isdir(directory_path):
        return {"error": "Invalid or missing directory path."}
    
    file_count = 0
    total_size = 0

    for root, _, files in os.walk(directory_path):
        file_count += len(files)
        total_size += sum(os.path.getsize(os.path.join(root, f)) for f in files if os.path.isfile(os.path.join(root, f)))

    return {"file_count": file_count, "total_size": total_size}

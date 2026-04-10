"""
Counter Tool Module
====================
A simple counter tool for the JARVIS system.
"""

from typing import Any, Dict

# Define the tool definition
TOOL_DEFINITION = {
    "name": "counter",
    "description": "A simple counter tool",
    "parameters": {
        "increment": {"type": "int", "default": 1},
        "reset": {"type": "bool", "default": False}
    }
}

# Initialize the counter
_counter = 0

def execute(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the counter tool.

    Args:
    - params (dict): A dictionary containing the tool parameters.
        - increment (int): The amount to increment the counter by. Defaults to 1.
        - reset (bool): Whether to reset the counter to 0. Defaults to False.

    Returns:
    - dict: A dictionary containing the updated counter value.
    """
    global _counter

    if params.get("reset", False):
        _counter = 0
    else:
        _counter += params.get("increment", 1)

    return {"count": _counter}

# Example usage
if __name__ == "__main__":
    print(execute({"increment": 5}))  # Output: {'count': 5}
    print(execute({"increment": 3}))  # Output: {'count': 8}
    print(execute({"reset": True}))  # Output: {'count': 0}

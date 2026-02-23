import pytest
import ast
from pathlib import Path

def test_app_py_syntax():
    """
    Checks src/app.py for syntax errors using ast.parse.
    """
    app_file_path = Path("src/app.py")
    assert app_file_path.exists(), f"File not found: {app_file_path}"

    try:
        with open(app_file_path, "r", encoding="utf-8") as f:
            source_code = f.read()
        ast.parse(source_code)
    except SyntaxError as e:
        pytest.fail(f"SyntaxError in {app_file_path}: {e}")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred while parsing {app_file_path}: {e}")


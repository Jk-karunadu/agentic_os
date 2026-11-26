import pytest
from pathlib import Path
from app.tools.list_directory import ListDirectoryTool


def test_list_directory_basic(tmp_path: Path):
    (tmp_path / "file1.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "file2.py").write_text("print('test')", encoding="utf-8")
    sub_dir = tmp_path / "subfolder"
    sub_dir.mkdir()
    (sub_dir / "nested.txt").write_text("nested", encoding="utf-8")

    tool = ListDirectoryTool()

    result = tool.execute(path=str(tmp_path))
    assert "file1.txt" in result
    assert "file2.py" in result
    assert "subfolder" in result
    assert "nested.txt" not in result


def test_list_directory_recursive(tmp_path: Path):
    (tmp_path / "root.txt").write_text("root", encoding="utf-8")
    sub_dir = tmp_path / "projects" / "my_app"
    sub_dir.mkdir(parents=True)
    (sub_dir / "main.py").write_text("print(1)", encoding="utf-8")

    tool = ListDirectoryTool()
    result = tool.execute(path=str(tmp_path), recursive=True)
    assert "root.txt" in result
    assert "projects" in result
    assert "my_app" in result
    assert "main.py" in result


def test_list_directory_pattern_filter(tmp_path: Path):
    (tmp_path / "test.py").write_text("code", encoding="utf-8")
    (tmp_path / "data.csv").write_text("1,2,3", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("notes", encoding="utf-8")

    tool = ListDirectoryTool()
    result = tool.execute(path=str(tmp_path), pattern="*.py")
    assert "test.py" in result
    assert "data.csv" not in result
    assert "notes.txt" not in result


def test_list_directory_invalid_path():
    tool = ListDirectoryTool()
    result = tool.execute(path="Z:\\non_existent_folder_xyz_999")
    assert "does not exist" in result or "Error" in result


def test_list_directory_missing_path():
    tool = ListDirectoryTool()
    result = tool.execute()
    assert "required" in result.lower()

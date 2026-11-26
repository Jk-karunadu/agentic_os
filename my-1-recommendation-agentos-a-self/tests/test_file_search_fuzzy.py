import pytest
from pathlib import Path
from app.tools.file_search import FileSearchTool


def test_file_search_fuzzy_filename(tmp_path: Path):
    proj_dir = tmp_path / "projects"
    proj_dir.mkdir()
    (proj_dir / "Face_Recognition_Model.py").write_text("# face recognition code", encoding="utf-8")

    tool = FileSearchTool()
    
    # Query with typo: "Face_recognisation"
    result = tool.execute(query="Face_recognisation", search_path=str(proj_dir))
    assert "Face_Recognition_Model.py" in result


def test_file_search_folder_detection(tmp_path: Path):
    proj_dir = tmp_path / "projects"
    proj_dir.mkdir()
    face_folder = proj_dir / "Face_Recognition"
    face_folder.mkdir()
    (face_folder / "app.py").write_text("print('app')", encoding="utf-8")

    tool = FileSearchTool()
    
    # Query matching folder name
    result = tool.execute(query="face recognisation", search_path=str(proj_dir))
    assert "Face_Recognition" in result
    assert "[DIR]" in result or "folder" in result.lower()


def test_file_search_no_false_positives(tmp_path: Path):
    """Ensure unrelated files and single-letter directories do NOT match 'face recognition'."""
    proj_dir = tmp_path / "projects"
    proj_dir.mkdir()
    (proj_dir / "agentic_os").mkdir()
    (proj_dir / "config.py").write_text("setting = 1", encoding="utf-8")
    (proj_dir / "verification.py").write_text("verify = 1", encoding="utf-8")
    (proj_dir / "i").mkdir()
    (proj_dir / "c").mkdir()
    (proj_dir / "io").mkdir()

    tool = FileSearchTool()
    result = tool.execute(query="face recognition", search_path=str(proj_dir))
    assert "agentic_os" not in result
    assert "config.py" not in result
    assert "verification.py" not in result
    assert "No files or folders found" in result


def test_fuzzy_match_unit():
    fm = FileSearchTool._fuzzy_match
    # Valid matches
    assert fm("Face Recognition", "Face_recognisation")
    assert fm("Face Recognition", "Face_Recognition")
    assert fm("face recognition", "my_face_recognition_project")
    assert fm("whatsapp", "5319275A.WhatsAppDesktop_2.2636.100.0_x64__cv1g1gvanyjgm")

    # False positive rejections
    assert not fm("face recognition", "agentic_os")
    assert not fm("face recognition", "config.py")
    assert not fm("face recognition", "verification.py")
    assert not fm("face recognition", "i")
    assert not fm("face recognition", "c")
    assert not fm("face recognition", "io")

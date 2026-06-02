"""IO utility regression tests."""

import time

import pytest

from florence_forge.utils.io import (
    FileManager,
    append_text_file,
    copy_directory,
    copy_file,
    create_backup,
    ensure_dir,
    format_file_size,
    get_file_size,
    list_files,
    load_json,
    load_pickle,
    load_yaml,
    read_text_file,
    safe_write,
    save_json,
    save_pickle,
    save_yaml,
    write_text_file,
)


# ---------------------------------------------------------------------------
# ensure_dir
# ---------------------------------------------------------------------------


def test_ensure_dir_creates_nested(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert result.exists() and result.is_dir()


# ---------------------------------------------------------------------------
# json / yaml / pickle roundtrips
# ---------------------------------------------------------------------------


def test_json_roundtrip_creates_parent(tmp_path):
    path = tmp_path / "nested" / "data.json"
    save_json({"a": 1, "b": [1, 2]}, path)
    assert load_json(path) == {"a": 1, "b": [1, 2]}


def test_load_json_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="JSON文件不存在"):
        load_json(tmp_path / "missing.json")


def test_yaml_roundtrip(tmp_path):
    path = tmp_path / "cfg.yaml"
    save_yaml({"name": "测试", "values": [1, 2, 3]}, path)
    assert load_yaml(path) == {"name": "测试", "values": [1, 2, 3]}


def test_load_yaml_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="YAML文件不存在"):
        load_yaml(tmp_path / "missing.yaml")


def test_pickle_roundtrip(tmp_path):
    path = tmp_path / "obj.pkl"
    save_pickle({"x": (1, 2, 3)}, path)
    assert load_pickle(path) == {"x": (1, 2, 3)}


def test_load_pickle_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Pickle文件不存在"):
        load_pickle(tmp_path / "missing.pkl")


# ---------------------------------------------------------------------------
# copy_file / copy_directory
# ---------------------------------------------------------------------------


def test_copy_file_overwrite_semantics(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("hello", encoding="utf-8")
    dst = tmp_path / "out" / "dst.txt"

    copy_file(src, dst)
    assert dst.read_text(encoding="utf-8") == "hello"

    with pytest.raises(FileExistsError):
        copy_file(src, dst)

    src.write_text("updated", encoding="utf-8")
    copy_file(src, dst, overwrite=True)
    assert dst.read_text(encoding="utf-8") == "updated"


def test_copy_file_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError, match="源文件不存在"):
        copy_file(tmp_path / "nope.txt", tmp_path / "dst.txt")


def test_copy_directory_overwrite(tmp_path):
    src = tmp_path / "srcdir"
    src.mkdir()
    (src / "f.txt").write_text("1", encoding="utf-8")
    dst = tmp_path / "dstdir"

    copy_directory(src, dst)
    assert (dst / "f.txt").read_text(encoding="utf-8") == "1"

    with pytest.raises(FileExistsError):
        copy_directory(src, dst)

    (src / "f.txt").write_text("2", encoding="utf-8")
    copy_directory(src, dst, overwrite=True)
    assert (dst / "f.txt").read_text(encoding="utf-8") == "2"


def test_copy_directory_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError, match="源目录不存在"):
        copy_directory(tmp_path / "nope", tmp_path / "dst")


# ---------------------------------------------------------------------------
# file size
# ---------------------------------------------------------------------------


def test_get_file_size(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"12345")
    assert get_file_size(path) == 5


def test_get_file_size_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="文件不存在"):
        get_file_size(tmp_path / "missing")


@pytest.mark.parametrize(
    "size,expected",
    [(0, "0B"), (512, "512.0B"), (1024, "1.0KB"), (1024**2, "1.0MB"), (1024**3, "1.0GB")],
)
def test_format_file_size(size, expected):
    assert format_file_size(size) == expected


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------


def test_list_files_glob_and_recursive(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("y", encoding="utf-8")

    flat = list_files(tmp_path, "*.txt", recursive=False)
    assert len(flat) == 1

    deep = list_files(tmp_path, "*.txt", recursive=True)
    assert len(deep) == 2


def test_list_files_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="目录不存在"):
        list_files(tmp_path / "nope")


# ---------------------------------------------------------------------------
# backup / safe_write / text files
# ---------------------------------------------------------------------------


def test_create_backup_with_timestamp(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("v1", encoding="utf-8")
    backup = create_backup(path, timestamp=True)
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == "v1"
    assert backup.name != path.name


def test_create_backup_no_timestamp_to_custom_dir(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("v1", encoding="utf-8")
    backup_dir = tmp_path / "backups"
    backup = create_backup(path, backup_dir=backup_dir, timestamp=False)
    assert backup.parent == backup_dir
    assert backup.name == "data_backup.txt"


def test_create_backup_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="文件不存在"):
        create_backup(tmp_path / "missing.txt")


def test_safe_write_creates_backup_when_exists(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("old", encoding="utf-8")
    safe_write("new", path, backup=True)
    assert path.read_text(encoding="utf-8") == "new"
    backups = list(tmp_path.glob("doc_*.txt"))
    assert len(backups) == 1


def test_read_write_append_text_file(tmp_path):
    path = tmp_path / "deep" / "note.txt"
    write_text_file("hello", path)
    assert read_text_file(path) == "hello"

    append_text_file(" world", path)
    assert read_text_file(path) == "hello world"


def test_write_text_file_with_backup(tmp_path):
    path = tmp_path / "note.txt"
    write_text_file("first", path)
    write_text_file("second", path, backup=True)
    assert read_text_file(path) == "second"
    assert list(tmp_path.glob("note_*.txt"))


def test_read_text_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="文件不存在"):
        read_text_file(tmp_path / "missing.txt")


# ---------------------------------------------------------------------------
# FileManager
# ---------------------------------------------------------------------------


def test_file_manager_json_pickle_paths(tmp_path):
    fm = FileManager(tmp_path / "workspace")

    json_path = fm.save_json({"k": "v"}, "sub", "data.json")
    assert fm.exists("sub", "data.json")
    assert fm.load_json("sub", "data.json") == {"k": "v"}
    assert json_path.parent.name == "sub"

    fm.save_pickle([1, 2, 3], "obj.pkl")
    assert fm.load_pickle("obj.pkl") == [1, 2, 3]


def test_file_manager_ensure_dir_and_list(tmp_path):
    fm = FileManager(tmp_path / "ws")
    fm.ensure_dir("logs")
    fm.save_json({"a": 1}, "logs", "a.json")
    found = fm.list_files("*.json", recursive=True)
    assert any(p.name == "a.json" for p in found)


def test_file_manager_cleanup_old_files(tmp_path):
    fm = FileManager(tmp_path / "ws")
    old = fm.save_json({"a": 1}, "old.json")
    # backdate mtime to 30 days ago
    old_time = time.time() - 30 * 86400
    import os

    os.utime(old, (old_time, old_time))

    preview = fm.cleanup_old_files("*.json", days=7, dry_run=True)
    assert old in preview
    assert old.exists()  # dry run does not delete

    deleted = fm.cleanup_old_files("*.json", days=7, dry_run=False)
    assert old in deleted
    assert not old.exists()

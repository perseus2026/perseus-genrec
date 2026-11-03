"""File selection belongs to the caller of the shared hashing primitive."""

import hashlib

from genrecsys.utils.files import code_provenance


def test_provenance_hashes_only_explicit_files(tmp_path):
    source = tmp_path / "source.py"
    config = tmp_path / "recipe.yaml"
    source.write_text("value = 1\n")
    config.write_text("seed: 42\n")
    ignored = tmp_path / "ignored.py"
    ignored.write_text("value = 2\n")
    actual = code_provenance(tmp_path, iter([source, config]))
    assert actual["files"] == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (source, config)
    }
    assert actual == code_provenance(tmp_path, [config, source])
    ignored.write_text("value = 3\n")
    assert actual == code_provenance(tmp_path, [source, config])
    source.write_text("value = 4\n")
    assert code_provenance(tmp_path, [source, config])["sha256"] != actual["sha256"]

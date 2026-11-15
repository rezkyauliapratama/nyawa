import hashlib
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bench


class ContainerExecutionTests(unittest.TestCase):
    def test_default_prefix_targets_dashboard_container(self):
        prefix, mode = bench.build_cli_prefix(None, "nyawa-dashboard", "/usr/local/bin/nyawa")
        self.assertEqual(prefix, ["docker", "exec", "nyawa-dashboard", "/usr/local/bin/nyawa"])
        self.assertEqual(mode, "container")

    def test_explicit_binary_is_local_candidate(self):
        prefix, mode = bench.build_cli_prefix("./nyawa-candidate", "nyawa-dashboard", "/usr/local/bin/nyawa")
        self.assertEqual(prefix, ["./nyawa-candidate"])
        self.assertEqual(mode, "local")

    def test_container_identity_hashes_binary_inside_container(self):
        digest = "a" * 64
        with patch.object(bench, "run_checked", side_effect=["nyawa v1.2.6", f"{digest}  /usr/local/bin/nyawa"]) as run:
            version, actual_digest = bench.identify_binary(
                ["docker", "exec", "nyawa-dashboard", "/usr/local/bin/nyawa"],
                "container", None, "nyawa-dashboard", "/usr/local/bin/nyawa",
            )
        self.assertEqual(version, "nyawa v1.2.6")
        self.assertEqual(actual_digest, digest)
        self.assertEqual(run.call_args_list[1].args[0], [
            "docker", "exec", "nyawa-dashboard", "sha256sum", "/usr/local/bin/nyawa",
        ])

    def test_local_candidate_identity_hashes_local_file(self):
        payload = b"candidate binary"
        with tempfile.NamedTemporaryFile() as candidate:
            candidate.write(payload)
            candidate.flush()
            with patch.object(bench, "run_checked", return_value="nyawa candidate"):
                version, digest = bench.identify_binary(
                    [candidate.name], "local", candidate.name,
                    "nyawa-dashboard", "/usr/local/bin/nyawa",
                )
        self.assertEqual(version, "nyawa candidate")
        self.assertEqual(digest, hashlib.sha256(payload).hexdigest())

    def test_recall_command_uses_container_database_path(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=b"#1 [0.9] matched memory\n", stderr=b"",
        )
        prefix = ["docker", "exec", "nyawa-dashboard", "/usr/local/bin/nyawa"]
        with patch.object(bench.subprocess, "run", return_value=completed) as run:
            result = bench.run_recall(prefix, "/opt/data/.nyawa/memory.db", "query text", "hermes", 10)
        self.assertEqual(run.call_args.args[0], [
            *prefix, "recall", "/opt/data/.nyawa/memory.db", "query text", "--ns", "hermes",
        ])
        self.assertTrue(result["ok"])
        self.assertEqual(result["results"][0]["content"], "matched memory")

    def test_database_snapshot_is_ephemeral_and_copies_hnsw(self):
        with tempfile.TemporaryDirectory() as root:
            source_db = Path(root) / "source.db"
            connection = sqlite3.connect(source_db)
            connection.execute("CREATE TABLE sample (value TEXT)")
            connection.execute("INSERT INTO sample VALUES ('source')")
            connection.commit()
            connection.close()
            source_hnsw = Path(f"{source_db}.hnsw")
            source_hnsw.write_bytes(b"hnsw-snapshot")

            with bench.database_snapshot(str(source_db), root) as (snapshot_db, has_hnsw):
                self.assertTrue(has_hnsw)
                self.assertNotEqual(snapshot_db, str(source_db))
                snapshot = sqlite3.connect(f"file:{snapshot_db}?mode=ro", uri=True)
                self.assertEqual(snapshot.execute("SELECT value FROM sample").fetchone()[0], "source")
                snapshot.close()
                self.assertEqual(Path(f"{snapshot_db}.hnsw").read_bytes(), b"hnsw-snapshot")

            self.assertFalse(Path(snapshot_db).exists())
            source = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
            self.assertEqual(source.execute("SELECT value FROM sample").fetchone()[0], "source")
            source.close()

    def test_hnsw_copy_falls_back_to_dashboard_container_on_permission_error(self):
        with tempfile.TemporaryDirectory() as root:
            source_db = Path(root) / "source.db"
            connection = sqlite3.connect(source_db)
            connection.execute("CREATE TABLE sample (value TEXT)")
            connection.commit()
            connection.close()
            Path(f"{source_db}.hnsw").write_bytes(b"protected-index")

            with patch.object(bench.shutil, "copy2", side_effect=PermissionError):
                with patch.object(bench, "run_checked") as run:
                    with bench.database_snapshot(str(source_db), root, "nyawa-dashboard") as (snapshot_db, has_hnsw):
                        self.assertTrue(has_hnsw)
                        self.assertTrue(Path(snapshot_db).exists())

            self.assertEqual(run.call_args.args[0][0:4], [
                "docker", "exec", "nyawa-dashboard", "cp",
            ])


if __name__ == "__main__":
    unittest.main()

import glob
import os
from pathlib import Path


class CodeMiner:
    """Your digital archaeologist - digs through code ruins"""

    def __init__(self, repo_path):
        self.repo_path = Path(repo_path)
        self.ignored_dirs = {
            ".git",
            "node_modules",
            "__pycache__",
            "dist",
            "build",
            "venv",
        }
        self.code_extensions = {
            ".py",
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".go",
            ".rs",
            ".rb",
            ".php",
            ".swift",
            ".kt",
            ".scala",
            ".html",
            ".css",
            ".scss",
            ".json",
            ".yml",
            ".yaml",
            ".md",
            ".sql",
            ".sh",
            ".dockerfile",
            ".tf",
            ".xml",
        }

    def is_code_file(self, filepath):
        """Is this file worth mining?"""
        return filepath.suffix.lower() in self.code_extensions

    def should_ignore(self, path):
        """Should we skip this directory?"""
        for part in path.parts:
            if part in self.ignored_dirs:
                return True
        return False

    def read_file_with_metadata(self, filepath):
        """Read a file and include where it came from"""
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            return {
                "path": str(filepath.relative_to(self.repo_path)),
                "content": content,
                "size": len(content),
                "language": filepath.suffix[1:] if filepath.suffix else "unknown",
            }
        except Exception as e:
            print(f"⚠️ Failed to read {filepath}: {e}")
            return None

    def mine_all(self):
        """The main excavation - returns all code files"""
        print(f"⛏️  Mining code from {self.repo_path}...")

        all_files = []
        for root, dirs, files in os.walk(self.repo_path):
            root_path = Path(root)

            # Skip ignored directories
            dirs[:] = [d for d in dirs if not self.should_ignore(root_path / d)]

            for file in files:
                filepath = root_path / file
                if self.is_code_file(filepath) and not self.should_ignore(filepath):
                    file_data = self.read_file_with_metadata(filepath)
                    if file_data:
                        all_files.append(file_data)

        print(f"🎉 Found {len(all_files)} code files!")
        return all_files


# Quick test
if __name__ == "__main__":
    miner = CodeMiner(".")  # Current directory
    files = miner.mine_all()
    for f in files[:5]:  # Show first 5
        print(f"📄 {f['path']} ({f['language']}) - {f['size']} chars")

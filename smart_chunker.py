import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


class SmartCodeChunker:
    """Slices code at natural boundaries (functions, classes, etc.)"""

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        # Different separators for different languages
        self.separators = [
            "\n\nclass ",  # Classes
            "\n\ndef ",  # Functions
            "\n\nasync def ",
            "\n\nfunction ",  # JS functions
            "\n\nconst ",
            "\n\nlet ",
            "\n\nvar ",
            "\n\ninterface ",  # TypeScript
            "\n\ntype ",
            "\n\npublic ",  # Java/C#
            "\n\nprivate ",
            "\n\nprotected ",
            "\n\nstruct ",  # Rust/Go
            "\n\nimpl ",
            "\n\nfn ",
            "\n\n// ",  # Comments
            "\n\n/*",
            "\n\n# ",  # Python comments/imports
            "\n\nimport ",
            "\n\nfrom ",
            "\n\n",  # Double newline as fallback
            "\n",  # Single newline as last resort
        ]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_with_context(self, code_file):
        """Chunk code while preserving file context"""
        chunks = self.splitter.split_text(code_file["content"])

        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            # Add metadata to each chunk
            enriched_chunks.append(
                {
                    "content": chunk,
                    "source_file": code_file["path"],
                    "language": code_file["language"],
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "full_path": f"{code_file['path']} (chunk {i + 1}/{len(chunks)})",
                }
            )

        return enriched_chunks

    def chunk_batch(self, code_files):
        """Chunk multiple files"""
        all_chunks = []
        for file in code_files:
            chunks = self.chunk_with_context(file)
            all_chunks.extend(chunks)
            print(f"📦 Split {file['path']} into {len(chunks)} chunks")

        print(f"🎁 Total chunks: {len(all_chunks)}")
        return all_chunks


# Test it
if __name__ == "__main__":
    test_file = {
        "path": "test.py",
        "content": """
import os

class UserService:
    def get_user(self, id):
        return {"id": id, "name": "Test"}

    def create_user(self, data):
        return {"id": 1, **data}

def main():
    service = UserService()
    print(service.get_user(1))
""",
        "language": "py",
    }

    chunker = SmartCodeChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.chunk_with_context(test_file)

    for i, chunk in enumerate(chunks):
        print(f"\n🔪 Chunk {i + 1}:")
        print(chunk["content"][:100] + "...")
        print(f"From: {chunk['source_file']}")

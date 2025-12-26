import os
import sys

from code_miner import CodeMiner
from query_engine import CodeQueryEngine
from smart_chunker import SmartCodeChunker
from vectorizer import CodeVectorizer


class CodeRAGOrchestrator:
    """The conductor of your code RAG symphony"""

    def __init__(self, repo_path, persist_db=True):
        self.repo_path = repo_path
        self.persist_db = persist_db
        self.vectorizer = None
        self.engine = None

    def setup(self, reindex=False):
        """Setup the entire system"""
        print("🎻 Setting up Code RAG System...")

        # Step 1: Mine code
        miner = CodeMiner(self.repo_path)
        code_files = miner.mine_all()

        if not code_files:
            print("❌ No code files found!")
            return False

        # Step 2: Chunk code
        # Larger chunks for better context (you have plenty of RAM)
        chunker = SmartCodeChunker(chunk_size=1500, chunk_overlap=200)
        chunks = chunker.chunk_batch(code_files)

        if not chunks:
            print("❌ No chunks created from code files!")
            return False

        # Step 3: Create embeddings and store
        # Using larger, more accurate model - your GPU can handle it!
        self.vectorizer = CodeVectorizer(
            model_name="sentence-transformers/all-mpnet-base-v2",  # Better quality
            persist_dir="./chroma_db",
            use_gpu=True  # Enable GPU acceleration
        )

        if reindex or self.vectorizer.get_stats()["total_chunks"] == 0:
            print("🔄 Creating new embeddings...")
            self.vectorizer.embed_and_store(chunks)
        else:
            print("📂 Using existing embeddings...")

        # Step 4: Initialize query engine
        try:
            self.engine = CodeQueryEngine(
                vectorizer=self.vectorizer,
                llm_backend="ollama",  # Change to "openai" if preferred
                model="codellama:13b",  # Larger model for better quality! You have the VRAM
            )
        except Exception as e:
            print(f"⚠️ Warning: Failed to initialize LLM: {e}")
            print("You can still use the vectorizer for searches, but LLM queries won't work.")
            return False

        print("✅ Setup complete!")
        return True

    def interactive_session(self):
        """Start interactive Q&A session"""
        if not self.engine:
            print("❌ Engine not initialized. Run setup() first.")
            return

        print("\n" + "=" * 60)
        print("🤖 CODE RAG ASSISTANT - INTERACTIVE MODE")
        print("=" * 60)
        print("Commands:")
        print("  /help     - Show this help")
        print("  /sources  - Toggle source display")
        print("  /chunks N - Change number of chunks (default: 5)")
        print("  /quit     - Exit")
        print("=" * 60)

        show_sources = True
        n_chunks = 5

        while True:
            try:
                user_input = input("\n❓ Question: ").strip()

                # Handle commands
                if user_input.startswith("/"):
                    cmd = user_input[1:].lower().split()

                    if cmd[0] == "quit" or cmd[0] == "exit":
                        print("👋 Goodbye!")
                        break
                    elif cmd[0] == "help":
                        print("Available commands: /help, /sources, /chunks N, /quit")
                    elif cmd[0] == "sources":
                        show_sources = not show_sources
                        print(f"📚 Sources display: {'ON' if show_sources else 'OFF'}")
                    elif cmd[0] == "chunks" and len(cmd) > 1:
                        try:
                            n_chunks = int(cmd[1])
                            print(f"🔍 Using {n_chunks} chunks for retrieval")
                        except:
                            print("❌ Invalid number")
                    continue

                if not user_input:
                    continue

                # Get answer
                result = self.engine.ask(user_input, n_chunks=n_chunks)

                # Display answer
                print("\n" + "=" * 60)
                print("🤖 Answer:")
                print(result["answer"])

                # Display sources if enabled
                if show_sources and result["sources"]:
                    print(f"\n📚 Top {len(result['sources'])} sources:")
                    for i, source in enumerate(result["sources"], 1):
                        print(f"  {i}. 📄 {source['file']}")
                        print(f"     🔍 Match: {source['similarity']:.2f}")
                        if i == 1:  # Show preview of top result
                            print(f"     👁️  Preview: {source['preview']}")

                print("=" * 60)

            except KeyboardInterrupt:
                print("\n\n🛑 Interrupted. Type /quit to exit.")
            except Exception as e:
                print(f"❌ Error: {e}")

    def query_one(self, question):
        """Ask a single question and return result"""
        if not self.engine:
            print("❌ Engine not initialized.")
            return None

        return self.engine.ask(question)


# Main execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Code RAG Assistant")
    parser.add_argument("--repo", default=".", help="Path to code repository")
    parser.add_argument("--reindex", action="store_true", help="Force reindexing")
    parser.add_argument("--question", help="Ask a single question")

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = CodeRAGOrchestrator(args.repo)

    if orchestrator.setup(reindex=args.reindex):
        if args.question:
            # Single question mode
            result = orchestrator.query_one(args.question)
            if result:
                print("\n🤖 Answer:")
                print(result["answer"])
                if result["sources"]:
                    print(f"\n📚 Sources:")
                    for source in result["sources"]:
                        print(
                            f"  • {source['file']} (match: {source['similarity']:.2f})"
                        )
        else:
            # Interactive mode
            orchestrator.interactive_session()

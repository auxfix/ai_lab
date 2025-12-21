#!/usr/bin/env python3
"""
Test script to verify the RAG system setup
Run this after installing dependencies to check if everything works
"""

import sys


def test_imports():
    """Test if all required packages are installed"""
    print("🧪 Testing imports...")
    errors = []

    packages = [
        ("sentence_transformers", "SentenceTransformer"),
        ("chromadb", None),
        ("langchain.text_splitter", "RecursiveCharacterTextSplitter"),
        ("ollama", None),
        ("streamlit", None),
        ("numpy", None),
    ]

    for package, class_name in packages:
        try:
            if class_name:
                exec(f"from {package} import {class_name}")
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError as e:
            print(f"  ❌ {package}: {e}")
            errors.append(package)

    return errors


def test_module_imports():
    """Test if local modules import correctly"""
    print("\n🧪 Testing local modules...")
    errors = []

    modules = [
        "code_miner",
        "smart_chunker",
        "vectorizer",
        "query_engine",
        "main",
    ]

    for module in modules:
        try:
            __import__(module)
            print(f"  ✅ {module}.py")
        except Exception as e:
            print(f"  ❌ {module}.py: {e}")
            errors.append(module)

    return errors


def test_ollama():
    """Test Ollama connection"""
    print("\n🧪 Testing Ollama connection...")
    try:
        import ollama

        models = ollama.list()
        print("  ✅ Ollama is running")
        print(f"  📦 Available models: {len(models.get('models', []))}")

        # Check for recommended models
        model_names = [m["name"] for m in models.get("models", [])]
        recommended = ["codellama:7b", "llama3:8b", "mistral:7b"]

        for model in recommended:
            if any(model in name for name in model_names):
                print(f"  ✅ Found {model}")
                return True

        print(
            "  ⚠️  No recommended models found. Pull one with: ollama pull codellama:7b"
        )
        return False

    except Exception as e:
        print(f"  ❌ Ollama error: {e}")
        print("  💡 Start Ollama with: ollama serve")
        return False


def main():
    print("=" * 60)
    print("🔍 RAG System Test Suite")
    print("=" * 60)

    # Test package imports
    import_errors = test_imports()

    # Test local modules
    module_errors = test_module_imports()

    # Test Ollama
    ollama_ok = test_ollama()

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)

    if import_errors:
        print(f"❌ Missing packages: {', '.join(import_errors)}")
        print("   Fix: pip install -r requirements.txt")
    else:
        print("✅ All packages installed correctly")

    if module_errors:
        print(f"❌ Module errors: {', '.join(module_errors)}")
    else:
        print("✅ All local modules working")

    if ollama_ok:
        print("✅ Ollama ready")
    else:
        print("⚠️  Ollama needs attention")

    print("=" * 60)

    if not import_errors and not module_errors and ollama_ok:
        print("\n🎉 System is ready! Run: python main.py --repo .")
        return 0
    else:
        print("\n⚠️  Please fix the issues above before running the system")
        return 1


if __name__ == "__main__":
    sys.exit(main())


import json
from typing import Dict, List, Optional


class CodeQueryEngine:
    """Query engine that combines retrieval with LLM generation"""

    def __init__(self, vectorizer, llm_backend="ollama", model="codellama:7b"):
        self.vectorizer = vectorizer
        self.llm_backend = llm_backend
        self.model = model
        self._llm_client = None

        # Initialize LLM client based on backend
        self._init_llm()

    def _init_llm(self):
        """Initialize the LLM client"""
        if self.llm_backend == "ollama":
            try:
                import ollama

                self._llm_client = ollama
                # Test connection
                try:
                    self._llm_client.list()
                    print(f"✅ Connected to Ollama with model: {self.model}")
                except Exception as e:
                    print(f"⚠️ Ollama connection issue: {e}")
                    print("Make sure Ollama is running: ollama serve")
            except ImportError:
                print("❌ Ollama package not installed. Install with: pip install ollama")
                raise

        elif self.llm_backend == "openai":
            try:
                import openai

                self._llm_client = openai
                print(f"✅ OpenAI client initialized with model: {self.model}")
            except ImportError:
                print("❌ OpenAI package not installed. Install with: pip install openai")
                raise
        else:
            raise ValueError(f"Unsupported LLM backend: {self.llm_backend}")

    def _format_context(self, retrieved_chunks: List[Dict]) -> str:
        """Format retrieved code chunks into context string"""
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(f"### Code Snippet {i} (from {chunk['file']}):\n")
            context_parts.append(f"```{chunk['language']}\n{chunk['content']}\n```\n")

        return "\n".join(context_parts)

    def _build_prompt(self, question: str, context: str) -> str:
        """Build the prompt for the LLM"""
        prompt = f"""You are a helpful code assistant. Answer the user's question based on the provided code context.

Context from codebase:
{context}

User Question: {question}

Instructions:
- Answer based on the code snippets provided above
- Be specific and reference actual code when relevant
- If the context doesn't contain enough information, say so
- Provide code examples when helpful

Answer:"""
        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and get response"""
        try:
            if self.llm_backend == "ollama":
                response = self._llm_client.generate(
                    model=self.model,
                    prompt=prompt,
                    options={
                        "temperature": 0.7,
                        "num_predict": 500,
                    },
                )
                return response["response"]

            elif self.llm_backend == "openai":
                response = self._llm_client.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful code assistant.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=500,
                )
                return response.choices[0].message.content

        except Exception as e:
            return f"Error calling LLM: {str(e)}"

    def ask(self, question: str, n_chunks: int = 5) -> Dict:
        """
        Ask a question about the codebase

        Args:
            question: The question to ask
            n_chunks: Number of relevant code chunks to retrieve

        Returns:
            Dictionary with 'answer' and 'sources'
        """
        try:
            # Step 1: Retrieve relevant code chunks
            results = self.vectorizer.search_similar(question, n_results=n_chunks)

            if not results["documents"][0]:
                return {
                    "answer": "No relevant code found in the repository.",
                    "sources": [],
                }

            # Step 2: Parse results
            retrieved_chunks = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                retrieved_chunks.append(
                    {
                        "content": doc,
                        "file": meta["source_file"],
                        "language": meta["language"],
                        "similarity": 1 - dist,  # Convert distance to similarity
                        "preview": doc[:150] + "..." if len(doc) > 150 else doc,
                    }
                )

            # Step 3: Format context
            context = self._format_context(retrieved_chunks)

            # Step 4: Build prompt
            prompt = self._build_prompt(question, context)

            # Step 5: Get LLM response
            answer = self._call_llm(prompt)

            return {"answer": answer.strip(), "sources": retrieved_chunks}

        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
            }


# Test the query engine
if __name__ == "__main__":
    from vectorizer import CodeVectorizer

    # Initialize vectorizer
    vectorizer = CodeVectorizer()

    # Initialize query engine
    engine = CodeQueryEngine(
        vectorizer=vectorizer, llm_backend="ollama", model="codellama:7b"
    )

    # Test query
    result = engine.ask("How does the code vectorizer work?")
    print("\n🤖 Answer:")
    print(result["answer"])

    if result["sources"]:
        print("\n📚 Sources:")
        for i, source in enumerate(result["sources"], 1):
            print(f"{i}. {source['file']} (similarity: {source['similarity']:.2f})")

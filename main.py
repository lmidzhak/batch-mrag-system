import os

from vector_loader import load_vectorstore
from rag_chain import get_rag_chain
from dotenv import load_dotenv

load_dotenv()


def run_query(query: str):
    os.environ["OPENAI_API_KEY"]

    vectorstore = load_vectorstore()
    rag_chain = get_rag_chain(vectorstore)

    result = rag_chain.invoke({"query": query})

    print("\n🧠 GPT-4 Answer:\n", result["result"])
    print("\n📚 Sources:")
    for doc in result["source_documents"]:
        print("—", doc.metadata.get("source"))


if __name__ == "__main__":
    run_query("What do you know about DEFIANCE Act?")

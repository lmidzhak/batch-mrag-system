from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI


def get_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", k=5)
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

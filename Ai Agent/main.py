from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# LLM Model
model = OllamaLLM(model="llama3.2")

# Prompt template
template = """
You are an expert in answering questions about Universities.

Here are some relevant reviews:
{reviews}

Here is the question to answer:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("------------------------")
    input_text = input("Enter your question: (q to quit) ")
    print("------------------------")

    if input_text.lower() == "q":
        break

    # Retriever’den ilgili dokümanları getir
    docs = retriever.invoke(input_text)

    # Dokümanların içeriklerini stringe çevir
    reviews_text = "\n".join([d.page_content for d in docs])

    # LLM’e gönder
    result = chain.invoke({"reviews": reviews_text, "question": input_text})

    print("Result:", result)

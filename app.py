import json
import os
import pinecone
import time
import pandas as pd
import requests
from dotenv import load_dotenv
import streamlit as st
from typing import List
import numpy as np
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
#
hf_token = os.getenv('HF_TOKEN')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_env = os.getenv('PINECONE_ENV')


def get_pinecone_index():
    # get API key from app.pinecone.io and environment from console
    pinecone.init(
        api_key=pinecone_api_key,
        environment=pinecone_env
    )

    index_name = 'llama2-rag'

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            index_name,
            dimension=384,
            metric='cosine'
        )
        # wait for index to finish initialization
        while not pinecone.describe_index(index_name).status['ready']:
            time.sleep(1)

        loader = PyPDFLoader("48lawsofpower.pdf")
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        docs = text_splitter.split_documents(data)
        docs = [doc.page_content for doc in docs]
        df = pd.DataFrame(docs, columns=["text"])

        from tqdm.auto import tqdm

        batch_size = 2  # can increase but needs larger instance size otherwise instance runs out of memory
        vector_limit = 1000

        pdf_df = df[:vector_limit]
        index = pinecone.Index(index_name)

        for i in tqdm(range(0, len(pdf_df), batch_size)):
            # find end of batch
            i_end = min(i + batch_size, len(pdf_df))
            # create IDs batch
            ids = [str(x) for x in range(i, i_end)]
            # create metadata batch
            metadatas = [{'text': text} for text in pdf_df["text"][i:i_end]]
            # create embeddings
            texts = pdf_df["text"][i:i_end].tolist()
            embeddings = embed_docs(texts)
            # create records list for upsert
            records = zip(ids, embeddings, metadatas)
            # upsert to Pinecone
            index.upsert(vectors=records)

    index = pinecone.Index(index_name)

    return index


# hit the embedding model endpoint
def embed_docs(docs: List[str]):
    url = "https://ues5110i6e.execute-api.us-east-1.amazonaws.com/prod"
    payload = json.dumps({
        "inputs": docs,
    })

    out = requests.post(url, payload, headers={"content-type": 'application/json', 'x-api-key': ''})
    array = np.array(out.json())
    embeddings = np.mean(np.array(array), axis=1)
    return embeddings.tolist()


def construct_context(contexts: List[str]) -> str:
    chosen_sections = []

    for text in contexts:
        text = text.strip()
        chosen_sections.append(text)
    concatenated_doc = "\n".join(chosen_sections)
    return concatenated_doc


def rag_query(question: str) -> str:
    # create query vec
    query_vec = embed_docs(question)[0]
    # query pinecone
    index = get_pinecone_index()
    res = index.query(vector=query_vec, top_k=5, include_metadata=True)
    # get contexts
    contexts = [match.metadata['text'] for match in res.matches]
    # build the multiple contexts string
    context_str = construct_context(contexts=contexts)

    prompt_template = """Answer the following QUESTION in a detailed way based only on the BOOK
    given. Do not add any unnessary information. If you do not know the answer and the BOOK doesn't
    contain the answer truthfully say "I don't know".

    BOOK:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """

    text_input = prompt_template.replace("{context}", context_str).replace("{question}", question)
    # hit the llama2 endpoint
    url = "https://r9dgkrfhk7.execute-api.us-east-1.amazonaws.com/prod"
    payload = json.dumps({
        "inputs": text_input,
        "parameters": {
            "do_sample": True,
            "top_p": 0.5,
            "temperature": 0.1,
            "top_k": 50,
            "max_new_tokens": 1024,
            "stop": [
                "</s>"
            ]
        }
    })

    response = requests.post(url, payload, headers={"content-type": 'application/json', 'x-api-key': ''})
    answer = response.json()[0]['generated_text'].replace(text_input, '')
    return answer


if __name__ == "__main__":
    # Set page title and header
    st.set_page_config(page_title="PDF Chatbot", page_icon="🤖")
    st.title("Chat with a PDF using using Meta's Llama2 Model")

    # Add a description
    st.write(
        "This chatbot allows you to interact with a PDF document using Meta's Llama2 model. You can ask "
        "questions about the document."
    )

    # Create a text input field for the user to enter their message
    user_input = st.text_input("Your message:", placeholder="Ask a question or start a conversation...")

    # Add a button for the user to send their message
    send_button = st.button("Send")

    # Create a text area to display the chatbot's response
    chatbot_response = st.empty()

    # Define the function to send the user's message to the chatbot and display the response
    def send_message(user_input):
        # Use the Llama2 model to generate a response to the user's message
        response = rag_query(user_input)

        # Display the chatbot's response in the text area
        chatbot_response.markdown(response)


    # Send the user's message to the chatbot when the send button is clicked
    if send_button:
        send_message(user_input)


from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
import pinecone
import time
from torch import cuda, bfloat16
import transformers
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

hf_token = os.getenv('HF_TOKEN')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_env = os.getenv('PINECONE_ENV')


def get_hf_embedding():
    embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'batch_size': 32}
    )
    return embed_model


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
        embed_model = get_hf_embedding()
        docsearch = Pinecone.from_texts([t.page_content for t in docs], embed_model, index_name=index_name)

    index = pinecone.Index(index_name)

    return index


def get_llm():
    model_id = 'meta-llama/Llama-2-13b-chat-hf'

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    # begin initializing HF items, need auth token for these
    hf_auth = hf_token
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=hf_auth
    )
    model.eval()
    print(f"Model loaded on {device}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task='text-generation',
        # we pass model parameters here too
        temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=512,  # mex number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
    )

    llm = HuggingFacePipeline(pipeline=generate_text)

    return llm


def get_response(question):
    text_field = 'text'  # field in metadata that contains text content

    index = get_pinecone_index()
    embed_model = get_hf_embedding()
    llm = get_llm()
    vectorstore = Pinecone(
        index, embed_model.embed_query, text_field
    )

    rag_pipeline = RetrievalQA.from_chain_type(
        llm=llm, chain_type='stuff',
        retriever=vectorstore.as_retriever()
    )

    response = rag_pipeline(f"""
         Answer the question asked in a very detailed manner from the provided book. 
         If the answer is not found in the book say "The book does not contain information about this". 
         Do not provide wrong answers.
         Question: \n{question}\n
         Answer:
    """)
    return response['result']


def main():
    # Set page title and header
    st.set_page_config(page_title="PDF Chatbot", page_icon="🤖")
    st.title("Chat with a PDF using Google's Gemini Pro Model")

    # Add a description
    st.write(
        "This chatbot allows you to interact with a PDF document using Google's Gemini Pro model. You can ask "
        "questions about the document, or even have a conversation with it."
    )

    # Create a text input field for the user to enter their message
    user_input = st.text_input("Your message:", placeholder="Ask a question or start a conversation...")

    # Add a button for the user to send their message
    send_button = st.button("Send")

    # Create a text area to display the chatbot's response
    chatbot_response = st.empty()

    # Define the function to send the user's message to the chatbot and display the response
    def send_message(user_input):
        # Use the Google Gemini Pro model to generate a response to the user's message
        response = get_response(user_input)

        # Display the chatbot's response in the text area
        chatbot_response.markdown(response)

    # Send the user's message to the chatbot when the send button is clicked
    if send_button:
        send_message(user_input)


if __name__ == "__main__":
    main()

import os

from flask import Flask, request, jsonify
import openai
from langchain import OpenAI, ConversationChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationSummaryBufferMemory
import sys
from flask_cors import CORS, cross_origin
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import recognize, audio_file_directory, pdf_file_directory, write_to_file, should_create_index
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")
pinecone_api_env = os.getenv("PINECONE_API_ENV")

llm = OpenAI(
    model_name='text-davinci-003',
    temperature=0,
    max_tokens=512
)

conversation_with_summary = ConversationChain(
    llm=llm,
    verbose=True
)

pinecone.init(
    api_key=pinecone_key,  # find at app.pinecone.io
    environment=pinecone_api_env  # next to api key in console
)
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
index_name = "pinecone-index"
chain = load_qa_chain(llm, chain_type="stuff")

@app.route('/hello')
@cross_origin()
def hello_world():
    return 'Hello World!'


@app.route('/messages', methods=['POST'])
def create_message():
    message_list = request.get_json()['messages']
    message = request.get_json()['message']
    generate_title = request.get_json()['generateTitle']
    memory = ConversationSummaryBufferMemory(llm=OpenAI())
    for elem1, elem2 in zip(message_list[::2], message_list[1::2]):
        if elem1.get('input') is not None:
            memory.save_context(elem1, elem2)
        else:
            memory.save_context(elem2, elem1)
        memory.save_context(elem1, elem2)

    conversation_with_summary.memory = memory
    return jsonify({'text': conversation_with_summary.predict(input=message),
                    'title': conversation_with_summary.predict(
                        input="generate a title for this chat, maximum 5 words, without quotation marks")}) \
        if generate_title else jsonify({'text': conversation_with_summary.predict(input=message)})
    # return jsonify({'text': "raspund la intrebare",
    #                 'title': "titlu"}) if generateTitle else \
    #     jsonify({'text': "raspund la intrebare"})


@app.route('/transcribe', methods=['POST'])
@cross_origin()
def transcribe():
    if 'file' not in request.files:
        return 'No file found in request!', 400

    file = request.files['file']
    filename = file.filename
    file.save(audio_file_directory + file.filename)
    return recognize(filename)


@app.route('/pdf', methods=['POST'])
@cross_origin()
def pdf():
    if 'file' not in request.files:
        return 'No file found in request!', 400

    # save file
    message = request.form['message']
    file = request.files['file']
    filename = file.filename
    file.save(pdf_file_directory + file.filename)
    write_to_file(filename)
    loader = UnstructuredFileLoader(pdf_file_directory + filename)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    index_name = "pinecone-index"
    pinecone.init(
        api_key=pinecone_key,  # find at app.pinecone.io
        environment=pinecone_api_env  # next to api key in console
    )
    pinecone.delete_index(index_name)
    pinecone.create_index(index_name, dimension=1536)
    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)


    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = docsearch.similarity_search(message, include_metadata=True)
    text = chain.run(input_documents=docs, question=message)
    return text

@app.route('/messages/pdf', methods=['POST'])
@cross_origin()
def create_message_pdf():
    # save file
    message = request.get_json()['message']
    filename = request.get_json()['filename']
    if should_create_index(filename):
        loader = UnstructuredFileLoader(pdf_file_directory + filename)
        data = loader.load()
        texts = text_splitter.split_documents(data)
        write_to_file(filename)
        pinecone.delete_index(index_name)
        pinecone.create_index(index_name, dimension=1536)
        docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

        docs = docsearch.similarity_search(message, include_metadata=True)
        text = chain.run(input_documents=docs, question=message)
    else:
        loader = UnstructuredFileLoader(pdf_file_directory + filename)
        data = loader.load()
        texts = text_splitter.split_documents(data)
        docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

        docs = docsearch.similarity_search(message, include_metadata=True)
        text = chain.run(input_documents=docs, question=message)
    return text

@app.route('/test', methods=['POST'])
@cross_origin()
def test():
    write_to_file('fix')
    x = should_create_index('fix')

    return "da" if x else "nu"


def create_index(filename):
    pass


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
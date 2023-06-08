import os
import sys
import time

import openai
import pinecone
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from langchain import OpenAI, ConversationChain
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

from utils import recognize, audio_directory, pdf_directory, write_to_file, should_create_index

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
    api_key=pinecone_key,
    environment=pinecone_api_env
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
    memory = ConversationSummaryBufferMemory(llm=OpenAI())
    for elem1, elem2 in zip(message_list[::2], message_list[1::2]):
        if elem1.get('input') is not None:
            memory.save_context(elem1, elem2)
        else:
            memory.save_context(elem2, elem1)
        memory.save_context(elem1, elem2)

    conversation_with_summary.memory = memory
    return jsonify({'text': conversation_with_summary.predict(input=message)})


@app.route('/transcribe', methods=['POST'])
@cross_origin()
def transcribe():
    if 'file' not in request.files:
        return 'No file found in request!', 400

    file = request.files['file']
    filename = file.filename
    file.save(audio_directory + file.filename)
    return recognize(filename)


@app.route('/pdf-messages', methods=['POST'])
@cross_origin()
def create_pdf_file():
    start_global = time.perf_counter()
    start = time.perf_counter()
    if 'file' not in request.files:
        return 'No file found in request!', 400

    # save file
    message = request.form['message']
    file = request.files['file']
    filename = file.filename
    file.save(pdf_directory + file.filename)
    write_to_file(filename)
    loader = UnstructuredFileLoader(pdf_directory + filename)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} seconds(s), load data')
    start = time.perf_counter()
    if len(pinecone.list_indexes()) > 0:
        pinecone.delete_index(index_name)
    pinecone.create_index(index_name, dimension=1536)
    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} seconds(s), delete/create index')
    start = time.perf_counter()
    docs = docsearch.similarity_search(message, include_metadata=True)
    text = chain.run(input_documents=docs, question=message)
    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} seconds(s), similarity and chain run')
    finish_global = time.perf_counter()
    print(f'Finished in {round(finish_global - start_global, 2)} seconds(s)')
    return jsonify({'text': text})


@app.route('/pdf-messages/without-pdf', methods=['POST'])
@cross_origin()
def create_pdf_message():
    start_global = time.perf_counter()
    # save file
    message = request.get_json()['message']
    filename = request.get_json()['filename']
    loader = UnstructuredFileLoader(pdf_directory + filename)
    data = loader.load()
    texts = text_splitter.split_documents(data)

    if should_create_index(filename):
        if len(pinecone.list_indexes()) > 0:
            pinecone.delete_index(index_name)
        pinecone.create_index(index_name, dimension=1536)
    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
    docs = docsearch.similarity_search(message, include_metadata=True, k=4)
    text = chain.run(input_documents=docs, question=message)

    write_to_file(filename)
    finish_global = time.perf_counter()
    print(f'Finished in {round(finish_global - start_global, 2)} seconds(s)')
    return jsonify({'text': text})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

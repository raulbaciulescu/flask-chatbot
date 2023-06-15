import os
import sys

import openai
import pinecone
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from utils import recognize, audio_directory, write_last_pdf_from_pinecone_index, should_create_index, \
    save_file, get_documents, create_pinecone_index, match_with_documents, \
    save_messages_in_memory, run_llm_with_documents, predict

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
#
pinecone.init(
    api_key=pinecone_key,
    environment=pinecone_api_env
)

@app.route('/hello')
@cross_origin()
def hello_world():
    return 'Hello World!'


@app.route('/messages', methods=['POST'])
def create_message():
    messages = request.get_json()['messages']
    message = request.get_json()['message']
    memory = save_messages_in_memory(messages)
    response = predict(memory, message)

    return jsonify({'text': response})


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
    if 'file' not in request.files:
        return 'No file found in request!', 400
    message = request.form['message']
    file = request.files['file']
    save_file(file)
    documents = get_documents(file.filename)
    create_pinecone_index()
    found_documents = match_with_documents(documents, message)
    text = run_llm_with_documents(found_documents, message)
    write_last_pdf_from_pinecone_index(file.filename)

    return jsonify({'text': text})


@app.route('/pdf-messages/without-pdf', methods=['POST'])
@cross_origin()
def create_pdf_message():
    message = request.get_json()['message']
    filename = request.get_json()['filename']
    documents = get_documents(filename)
    if should_create_index(filename):
        create_pinecone_index()

    found_documents = match_with_documents(documents, message)
    text = run_llm_with_documents(found_documents, message)
    write_last_pdf_from_pinecone_index(filename)

    return jsonify({'text': text})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

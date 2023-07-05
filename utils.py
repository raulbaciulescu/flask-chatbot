import pinecone
import speech_recognition as sr
from langchain import OpenAI, ConversationChain
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import GCSFileLoader

from cloud_utils import upload_blob, project_name, bucket_name

audio_directory = 'audios\\'
# pdf_directory = 'pdf\\'
pdf_directory = 'pdf/'
pdf_last_filename = 'pdf\\last_pdf.txt'
index_name = "pinecone-index"
run_in_cloud = True


def recognize(filename):
    filename = audio_directory + filename
    r = sr.Recognizer()
    audio_file = sr.AudioFile(filename)

    with audio_file as source:
        audio = r.record(source)

    return {"text": r.recognize_google(audio, language='ro-RO')}


def should_create_index(filename):
    f = open(pdf_last_filename, 'r')
    filename_from_file = f.read()
    return filename != filename_from_file


def write_last_pdf_from_pinecone_index(filename):
    f = open(pdf_last_filename, 'w')
    f.write(filename)
    f.close()


def save_file(file):
    filename = file.filename
    file.save(pdf_directory + file.filename)
    if run_in_cloud:
        upload_blob(filename)
    write_last_pdf_from_pinecone_index(filename)


def get_documents(filename):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    if run_in_cloud:
        loader = GCSFileLoader(project_name=project_name, bucket=bucket_name, blob=filename)
        data = loader.load()
    else:
        loader = UnstructuredFileLoader(pdf_directory + filename)
        data = loader.load()

    return text_splitter.split_documents(data)


def create_pinecone_index():
    if len(pinecone.list_indexes()) > 0:
        pinecone.delete_index(index_name)
    pinecone.create_index(index_name, dimension=1536)


def match_with_documents(texts, message):
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
    docs = docsearch.similarity_search(message, include_metadata=True)
    return docs


def save_messages_in_memory(message_list):
    memory = ConversationSummaryBufferMemory(llm=OpenAI(), max_token_limit=2500)
    for elem1, elem2 in zip(message_list[::2], message_list[1::2]):
        if elem1.get('input') is not None:
            memory.save_context(elem1, elem2)
        else:
            memory.save_context(elem2, elem1)
        memory.save_context(elem1, elem2)

    return memory


def run_llm_with_documents(found_documents, message):
    llm = OpenAI(
        model_name='text-davinci-003',
        temperature=0,
    )
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain.run(input_documents=found_documents, question=message)


def predict(memory, message):
    summary_chain = ConversationChain(
        llm=OpenAI(model="text-davinci-003"),
        verbose=True
    )
    summary_chain.memory = memory
    return summary_chain.predict(input=message)

import speech_recognition as sr

audio_file_directory = 'D:\\Facultate\\Licenta\\projects\\flask-chatbot\\audios\\'
pdf_file_directory = 'D:\\Facultate\\Licenta\\projects\\flask-chatbot\\pdf\\'
pdf_last_filename = 'D:\\Facultate\\Licenta\\projects\\flask-chatbot\\pdf\\last_pdf.txt'
texts_directory = 'D:\\Facultate\\Licenta\\projects\\flask-chatbot\\texts\\'

def recognize(filename):
    filename = audio_file_directory + filename
    r = sr.Recognizer()
    romana = sr.AudioFile(filename)

    with romana as source:
        audio = r.record(source)

    return {"text": r.recognize_google(audio, language='ro-RO')}


def should_create_index(filename):
    f = open(pdf_last_filename, 'r')
    filename_from_file = f.read()
    return filename != filename_from_file


def write_to_file(filename):
    f = open(pdf_last_filename, 'w')
    f.write(filename)
    f.close()

from faster_whisper import WhisperModel
from docx import Document
from dotenv import load_dotenv
import os
from transcribe_audio import transcribe, save_transcript
from summarize_text import summarize_by_model, save_summary
load_dotenv()

MODEL_PATH = os.getenv('MODEL_PATH')
AUDIO_FILE = os.getenv('AUDIO_FILE')
TRANSCRIPT_FILE = os.getenv('TRANSCRIPT_FILE')
SUMMARY_FILE = os.getenv('SUMMARY_FILE')

def read_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return "\n".join(full_text)

if __name__ == "__main__":

    is_transcribe_input = False if input("Do you want to transcribe the audio ? give True or False") == "False" else True
    if (is_transcribe_input):
        audio_model = WhisperModel("small", device="cpu", compute_type="int8")


        print("Model initialized")
        print("Transcripting...")

        segments = transcribe(AUDIO_FILE, audio_model)
        transcript = save_transcript(segments, TRANSCRIPT_FILE)

    docx_modified = False if input("Did you modified the transcript given from the model ? give True or False") == "False" else True
    if (not is_transcribe_input or docx_modified):
        transcript = read_docx(TRANSCRIPT_FILE)

    chunk_summaries = summarize_by_model(transcript, "facebook/bart-large-cnn")
    save_summary(SUMMARY_FILE, chunk_summaries)







from faster_whisper import WhisperModel
from docx import Document
from dotenv import load_dotenv
import os
from transcribe_audio import transcribe, save_transcript
from summarize_text import t5_small, save_summary, bart
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

    audio_model = WhisperModel("small", device="cpu", compute_type="int8")


    print("Model initialized")
    print("Transcripting...")

    segments = transcribe(AUDIO_FILE, audio_model)
    transcript = save_transcript(segments, TRANSCRIPT_FILE)
    transcript = read_docx(TRANSCRIPT_FILE)

    chunk_summaries = bart(transcript)
    save_summary(SUMMARY_FILE, chunk_summaries)







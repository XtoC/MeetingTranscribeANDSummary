from faster_whisper import WhisperModel
from docx import Document
from dotenv import load_dotenv
import os
load_dotenv()

MODEL_PATH = os.getenv('MODEL_PATH')
AUDIO_FILE = os.getenv('AUDIO_FILE')
DOC_FILE = os.getenv('DOC_FILE')

if __name__ == "__main__":

    model = WhisperModel("small", device="cpu", compute_type="int8")

    print("Model initialized")
    print("Transcripting...")
    segments, _ = model.transcribe(AUDIO_FILE, beam_size=5)

    doc = Document()
    doc.add_heading("Transcription", level=1)

    # Add each segment
    for segment in segments:
        print(segment.text)
        doc.add_paragraph(segment.text)

    doc.save(DOC_FILE)
    print(f"Transcription saved to {DOC_FILE}")



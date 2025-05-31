from docx import Document
from faster_whisper import WhisperModel
import os

AUDIO_FILE = os.getenv('AUDIO_FILE')
TRANSCRIPT_FILE = os.getenv('TRANSCRIPT_FILE')
def transcribe(filename, model):
    segments, _ = model.transcribe(filename, beam_size=5)

    return segments

def save_transcript(segments, filename):
    doc = Document()
    doc.add_heading("Meeting Transcript", level=1)
    text = ""
    for seg in segments:
        # print(seg.text)
        doc.add_paragraph(seg.text)
        text += seg.text
    doc.save(filename)
    print(f"Transcript saved to {filename}")

    return text

if __name__ == "__main__"
    audio_model = WhisperModel("small", device="cpu", compute_type="int8")


    print("Model initialized")
    print("Transcripting...")

    segments = transcribe(AUDIO_FILE, audio_model)
    transcript = save_transcript(segments, TRANSCRIPT_FILE)
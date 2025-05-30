from docx import Document
from pydub import AudioSegment
import os

def split_audio(input_file, chunk_duration_ms):
    audio = AudioSegment.from_file(input_file)
    chunks = []
    total_length = len(audio)
    for i in range(0, total_length, chunk_duration_ms):
        chunk = audio[i:i + chunk_duration_ms]
        chunk_name = f"chunk_{i // chunk_duration_ms}.wav"
        chunk.export(chunk_name, format="wav")
        chunks.append(chunk_name)
    return chunks

def transcribe_chunks(chunks, model):
    all_segments = []
    for chunk in chunks:
        print(f"Transcribing {chunk}...")
        segments, _ = model.transcribe(chunk, beam_size=5)
        all_segments.extend(segments)
        os.remove(chunk)  # optional: clean up
    return all_segments

def save_transcript(segments, filename):
    doc = Document()
    doc.add_heading("Meeting Transcript", level=1)
    for seg in segments:
        doc.add_paragraph(seg.text)
    doc.save(filename)
    print(f"Transcript saved to {filename}")
from faster_whisper import WhisperModel
from docx import Document
from gpt4all import GPT4All

AUDIO_FILE = "./meeting_record.mp4"
DOC_FILE = "./transcription.docx"

def chunk_text(text, max_chunk_size=1000):
    """
    Split text into chunks roughly max_chunk_size tokens (approx words here).
    This naive split just cuts by words.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_chunk_size):
        chunk = " ".join(words[i:i + max_chunk_size])
        chunks.append(chunk)
    return chunks


def summarize_chunk(model, chunk, max_tokens=300):
    prompt = (
            "Please provide a concise summary of the following meeting transcript section:\n\n"
            + chunk
            + "\n\nSummary:"
    )
    summary = model.generate(prompt, max_tokens=max_tokens)
    return summary.strip()


def summarize_long_transcript(transcript, model_path, chunk_size=1000, max_tokens=300):
    # Load model
    gptj = GPT4All(model_path)

    chunks = chunk_text(transcript, max_chunk_size=chunk_size)
    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i + 1}/{len(chunks)}...")
        summary = summarize_chunk(gptj, chunk, max_tokens=max_tokens)
        chunk_summaries.append(summary)

    # Optionally combine all chunk summaries into a final summary
    final_prompt = (
            "The following are summaries of sections from a meeting transcript. "
            "Please provide an overall concise summary of the entire meeting:\n\n"
            + "\n\n".join(chunk_summaries)
            + "\n\nFinal Summary:"
    )
    final_summary = gptj.generate(final_prompt, max_tokens=max_tokens)

    return final_summary.strip(), chunk_summaries

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



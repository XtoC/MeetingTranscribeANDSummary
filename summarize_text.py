from gpt4all import GPT4All
from docx import Document
from transformers import pipeline
from main import read_docx
import os

TRANSCRIPT_FILE = os.getenv('TRANSCRIPT_FILE')
SUMMARY_FILE = os.getenv('SUMMARY_FILE')
def chunk_text(text, max_chunk_size=350):
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


def summarize_chunk(model, chunk, max_tokens=1000):
    prompt = (
            "Please provide a concise summary of the following meeting transcript section:\n\n"
            + chunk
            + "\n\nSummary:"
    )
    summary = model.generate(prompt, max_tokens=max_tokens)
    return summary.strip()


def summarize_long_transcript(transcript, model_path, chunk_size=1000, max_tokens=1000):
    # Load model
    gptj = GPT4All(model_name="DeepSeek-R1-Distill-Qwen-7B~Q4_0.gguf", model_path=model_path, allow_download=False)

    chunks = chunk_text(transcript, max_chunk_size=chunk_size)
    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i + 1}/{len(chunks)}...")
        summary = summarize_chunk(gptj, chunk, max_tokens=max_tokens)
        chunk_summaries.append(summary)

    # Optionally combine all chunk summaries into a final summary
    # final_prompt = (
    #         "The following are summaries of sections from a meeting transcript. "
    #         "Please provide an overall concise summary of the entire meeting:\n\n"
    #         + "\n\n".join(chunk_summaries)
    #         + "\n\nFinal Summary:"
    # )
    # final_summary = gptj.generate(final_prompt, max_tokens=max_tokens)

    return chunk_summaries

def bart(input_text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    chunk_transcript = chunk_text(input_text, 512)
    chunk_summaries = []
    for transcript in chunk_transcript:
        summary = summarizer(transcript, max_length=160, min_length=60, do_sample=False)
        chunk_summaries.append(summary[0]['summary_text'])

    return chunk_summaries

def t5_small(input_text):
    # Load the summarization pipeline using BART
    summarizer = pipeline("summarization", model="t5-base")

    # Example meeting transcript (shortened)
    chunk_transcript = chunk_text(input_text, 350)

    chunk_summaries = []
    # Run summarization
    for transcript in chunk_transcript:
        summary = summarizer(transcript, do_sample=False)
        chunk_summaries.append(summary[0]['summary_text'])

    return chunk_summaries

def save_summary(filename, chunk_summaries):
    doc = Document()
    doc.add_heading("Meeting Summary", level=1)
    for chunk in chunk_summaries:
        doc.add_paragraph(chunk)
    doc.save(filename)
    print(f"Summary saved to {filename}")

if __name__ == "__main__":
    transcript = read_docx(TRANSCRIPT_FILE)

    chunk_summaries = bart(transcript)
    save_summary(SUMMARY_FILE, chunk_summaries)
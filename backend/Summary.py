from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline

def get_video_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = [entry['text'] for entry in transcript]
        transcript_text = " ".join(text).replace("\n", " ")
        return transcript_text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_video_summary(video_id):
    transcript = get_video_transcript(video_id)
    
    summarizer = pipeline("summarization")
    max_chunk = 500

    # Split text into sentences at '<eos>'
    content = transcript.split(' ')

    current_chunk = []
    chunks = []

    for word in content:
        # If adding the current word doesn't exceed the word limit
        if sum(len(word.split()) for word in current_chunk) + len(word.split()) <= max_chunk:
            current_chunk.append(word)
        else:
            # If the current word puts the chunk over the limit, start a new chunk
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]

    # Add the last chunk
    chunks.append(' '.join(current_chunk))
    res = summarizer(chunks, max_length=120, min_length=30, do_sample=False)
    ' '.join([summ['summary_text'] for summ in res])
    text = ' '.join([summ['summary_text'] for summ in res])
    
    return text
    
video_id = "njKP3FqW3Sk"

print(get_video_summary(video_id))

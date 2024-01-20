from youtube_transcript_api import YouTubeTranscriptApi
from transformers import BartForConditionalGeneration, BartTokenizer

def get_video_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_video_summary(video_id):
    transcript = get_video_transcript(video_id)

    if transcript:
        transcript_text = " ".join(entry['text'] for entry in transcript)
        print(transcript_text)

        model_name = 'facebook/bart-large-cnn'
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name).to('cpu')  # Change 'cuda' to 'cpu' if you don't have a GPU

        inputs = tokenizer(transcript_text, max_length=1024, return_tensors='pt', truncation=True)
        summary_ids = model.generate(inputs['input_ids'], max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary
    else:
        return None

video_id = "njKP3FqW3Sk"

print(get_video_summary(video_id))

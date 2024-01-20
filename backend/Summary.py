from youtube_transcript_api import YouTubeTranscriptApi
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

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
        transcript_text = ""
        for entry in transcript:
            transcript_text += f"{entry['text']} "

        model_name = 'google/pegasus-xsum'
        torch_device = 'cpu'
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

        batch = tokenizer.prepare_seq2seq_batch(transcript_text, truncation=True, padding='longest', return_tensors="pt").to(torch_device)
        translated = model.generate(**batch)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)

        return tgt_text
    else:
        return None

video_id = "njKP3FqW3Sk"

print(get_video_summary(video_id))
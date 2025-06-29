import librosa
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification

LONG_PAUSE_THRESHOLD = 3.0
WPM_LOWER_THRESHOLD = 90
WPM_UPPER_THRESHOLD = 160
# FILLER_WORDS = ["um", "uh", "ah", "er", "hmm"]
EMOTION_SCORES = {
    "ang": 2.0,
    "sad": 3.0,
    "neu": 6.0,
    "hap": 10.0,
}
EMOTION_FULL = {
    "ang": "Angry",
    "sad": "Sad",
    "neu": "Neutral",
    "hap": "Happy"
}

def get_fluency_score(y, sr, transcription):
    duration_sec = librosa.get_duration(y=y, sr=sr)
    word_count = len(transcription.split())
    wpm = word_count / (duration_sec / 60) if duration_sec > 0 else 0

    # Detect long pauses using amplitude thresholding
    intervals = librosa.effects.split(y, top_db=30)
    long_pauses = 0
    total_pause_time = 0
    for i in range(1, len(intervals)):
        pause_duration = (intervals[i][0] - intervals[i-1][1]) / sr
        total_pause_time += pause_duration
        if pause_duration >= LONG_PAUSE_THRESHOLD:
            long_pauses += 1

    # Scoring logic
    fluency_score = 10.0
    if wpm < WPM_LOWER_THRESHOLD:
        fluency_score -= (WPM_LOWER_THRESHOLD - wpm) / 10  # 1 point per 10 WPM under WPM_LOWER_THRESHOLD
    elif wpm > WPM_UPPER_THRESHOLD:
        fluency_score -= (wpm - WPM_UPPER_THRESHOLD) / 10  # 1 point per 10 WPM over WPM_UPPER_THRESHOLD
    fluency_score -= min(2.0, long_pauses * 0.5)  # Max -2 for long pauses longer than LONG_PAUSE_THRESHOLD seconds
    fluency_score = round(max(0.0, min(10.0, fluency_score)), 1)

    feedback = f"{int(wpm)} WPM, {long_pauses} long pause(s) ≥ {LONG_PAUSE_THRESHOLD}s" 
    return fluency_score, feedback

# TODO: Does not accurately count filler words
# def count_filler_words(transcription):
#     transcription = transcription.lower()
#     return sum(transcription.count(word) for word in FILLER_WORDS)

def get_confidence_score(y, sr, transcription):
    volume = np.mean(np.abs(y))
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0

    # filler_count = count_filler_words(transcription)

    confidence_score = 10.0
    if volume < 0.01:
        confidence_score -= 3.0
    if pitch_std < 20:
        confidence_score -= 2.0
    # confidence_score -= min(5.0, filler_count * 0.5)  # Cap filler word penalty

    feedback = f"Volume: {volume:.3f}, Pitch Var: {pitch_std:.2f}"  # Filler Words: {filler_count}
    confidence_score = round(max(0.0, min(10.0, confidence_score)), 1)
    return confidence_score, feedback

def get_emotion_score(audio_path):
    processor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-base-superb-er")
    model = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-er")
    
    speech, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted = torch.argmax(logits, dim=-1).item()

    labels = model.config.id2label
    emotion = labels[predicted].lower()

    # Scoring
    emotion_score = EMOTION_SCORES.get(emotion, 5.0)  # Default to neutral score
    emotion_full = EMOTION_FULL.get(emotion, "Unknown")
    return round(emotion_score, 1), f"Detected Emotion: {emotion_full}"

def analyse_audio(audio_path, transcription_path):
    y, sr = librosa.load(audio_path)
    try:
        with open(transcription_path, 'r') as f:
            transcription = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Transcription file not found: {transcription_path}")
    
    fluency, fluency_feedback = get_fluency_score(y, sr, transcription)
    confidence, confidence_feedback = get_confidence_score(y, sr, transcription)
    emotion, emotion_feedback = get_emotion_score(audio_path)

    return {
        "Fluency": {"Score": fluency, "Feedback": fluency_feedback},
        "Confidence": {"Score": confidence, "Feedback": confidence_feedback},
        "Emotion": {"Score": emotion, "Feedback": emotion_feedback}
    }

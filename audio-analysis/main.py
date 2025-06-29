from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from audio_scoring import analyse_audio
import time
import os

AUDIO_DIR = "audio"
ALLOWED_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg")

app = FastAPI()

@app.post("/analyse-audio/")
async def analyse_audio_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Accepted: WAV, MP3, FLAC, OGG."
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    os.makedirs(AUDIO_DIR, exist_ok=True)
    file_path = os.path.join(AUDIO_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    transcription_path = os.path.join("..", "transcriber", "transcripts", f"{os.path.splitext(file.filename)[0]}_transcript.txt")

    try:
        start_time = time.time()
        results = analyse_audio(file_path, transcription_path)
        duration = time.time() - start_time

        response = {
            "results": {
                feature: {
                    "Score": data.get("Score"),
                    "Feedback": data.get("Feedback")
                } for feature, data in results.items()
            },
            "duration_seconds": round(duration, 2)
        }
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

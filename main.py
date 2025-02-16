from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from beat_this.inference import File2Beats

# Initialize FastAPI app
app = FastAPI()

# Load the model once to avoid reloading it on each request
file2beats = File2Beats(checkpoint_path="final0", dbn=False)

# Define request model
class AudioInput(BaseModel):
    signal: list  # Assuming signal is passed as a list of floats
    sr: int       # Sample rate

@app.post("/detect_beats")
def detect_beats(audio_input: AudioInput):
    try:
        # Convert list to NumPy array
        signal = np.array(audio_input.signal, dtype=np.float64)
        sr = audio_input.sr
        
        # Process the input to get beats and downbeats
        beats, downbeats = file2beats(signal, sr)
        
        return {"beats": beats.tolist(), "downbeats": downbeats.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

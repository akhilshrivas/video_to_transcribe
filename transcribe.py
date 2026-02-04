import os
from faster_whisper import WhisperModel
import time

# --- CONFIGURATION ---
# "small" is standard. If you need MAXIMUM accuracy and have time to wait, 
# change this to "medium". (Medium is slower but much smarter).
MODEL_SIZE = "small" 

print(f"Loading {MODEL_SIZE} model... (Processing will start after file selection)")
# Run on CPU with INT8 quantization
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

def transcribe_any_video():
    # 1. GET USER INPUT
    # This allows you to paste any filename or path when the script runs
    print("\n---------------------------------------------------")
    video_path = input("Enter the video filename (e.g., meeting.mp4): ").strip()
    
    # Remove quotes if user dragged and dropped file path
    video_path = video_path.replace('"', '').replace("'", "")

    if not os.path.exists(video_path):
        print(f" Error: The file '{video_path}' was not found. Check the name.")
        return

    print(f"---  Processing: {video_path} ---")
    start_time = time.time()

    # 2. ACCURACY PROMPT
    # We keep this generic for tech/coding environments
    description_prompt = (
        "Transcribe this technical discussion. It may contain mixed English and Hindi (Hinglish). "
        "Keywords: API, SQL, Database, Pipeline, Bug, Deployment, Error, Server, Testing."
    )

    # 3. TRANSCRIBE WITH HIGH ACCURACY SETTINGS
    segments, info = model.transcribe(
        video_path, 
        beam_size=5,       # CHANGED: 5 = Higher accuracy (slower). 1 = Fast (less accurate).
        vad_filter=True,   # Keep VAD to remove silence
        vad_parameters=dict(min_silence_duration_ms=500),
        initial_prompt=description_prompt,
        language=None      # Let model auto-detect (usually switches to 'hi' or 'en')
    )

    print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")

    full_transcript = []
    
    print("\n--- Transcript Start ---\n")
    for segment in segments:
        # Format timestamp as [MM:SS]
        timestamp = time.strftime('%M:%S', time.gmtime(segment.start))
        text = f"[{timestamp}] {segment.text}"
        
        print(text)
        full_transcript.append(text)

    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n---  Done in {duration:.2f} seconds ---")

    # 4. SAVE GENERICALLY
    # Saves as "yourvideo_transcript.txt" automatically
    base_name = os.path.basename(video_path)
    filename = f"{os.path.splitext(base_name)[0]}_transcript.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(full_transcript))
    print(f" Saved transcript to: {filename}")

# Run the function
if __name__ == "__main__":
    transcribe_any_video()
import whisper
from typing import Literal

# Literal type definitions
MODEL_SIZE = Literal["tiny", "base", "small", "medium", "large"]
LANGUAGE = Literal["en", "ja"]

# set context
MODEL_SIZE = "small"
LANGUAGE = "ja"

def main():
    # enum for whisper model size
    model = whisper.load_model(MODEL_SIZE)

    # file path to audio file
    audio_file = "audio.mp3"

    # fp16 is for use cpu
    result = model.transcribe(audio_file, verbose=True, language=LANGUAGE, fp16=False)

if __name__ == "__main__":
    main()
    
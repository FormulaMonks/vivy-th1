from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
import shutil
import os
from google.oauth2 import service_account
from constants import KEY_PATH
from speech_to_text import SpeechToText
from text_to_text import TextToText
from text_to_speech import GoogleCloudTTS, ElevenLabsTTS
from logger import logger
from pydantic import BaseModel
import asyncio

app = FastAPI()

class AudioInput(BaseModel):
    want_sound: bool
    use_eleven_labs: bool
    debug: bool

@app.post("/process-audio")
async def process_audio(input_data: AudioInput, audio_file: UploadFile = File(...), background_tasks: BackgroundTasks):
    try:
        # Save the audio file temporarily
        temp_file = f"temp_{audio_file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        # Process the audio file and get the response
        text_response = await asyncio.to_thread(run_processing, temp_file, input_data)

        # Schedule the removal of the temporary file
        background_tasks.add_task(os.remove, temp_file)

        # Return the response
        return JSONResponse(content={"response": text_response})

    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def run_processing(filename: str, input_data: AudioInput):
    credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
    s2t = SpeechToText(credentials)
    t2t = TextToText(messages=[{"role": "system", "content": "You are Vivy, the AI songstress..."}])

    t2s = ElevenLabsTTS() if input_data.use_eleven_labs else GoogleCloudTTS(credentials)

    if input_data.debug:
        return "Checking if debug mode works or not."

    print("Transcribing...")
    prompt = s2t.transcribe_audio(filename)
    print(prompt)
    text_response = t2t.generate_response(prompt)
    print(text_response)
    
    if input_data.want_sound:
        t2s.synthesize(text_response)

    return text_response

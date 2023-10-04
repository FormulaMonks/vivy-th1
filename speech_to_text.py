import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import logging
from google.cloud import speech
from constants import SAMPLE_RATE_HERTZ, LANGUAGE_CODE

logger = logging.getLogger(__name__)

class SpeechToText:
    def __init__(self, credentials):
        self.client = speech.SpeechClient(credentials=credentials)

    def record_audio(self, max_duration=10, chunk_duration=0.5, silence_duration=1.5, energy_ratio_threshold=1.5, initialization_chunks=2):
        try:
            print("Preparing to record audio...")
            
            chunk_size = int(SAMPLE_RATE_HERTZ * chunk_duration)
            recording = False
            audio_data = []
            silence_counter = 0
            start_time = time.time()

            # Initialize long_term_energy with the energy of a few initial silence chunks
            initial_chunks = []
            for _ in range(initialization_chunks):
                chunk = sd.rec(chunk_size, samplerate=SAMPLE_RATE_HERTZ, channels=1, dtype=np.int16, blocking=True)
                chunk = np.squeeze(chunk)
                initial_chunks.append(chunk)
            long_term_energy = np.sum(np.concatenate(initial_chunks).astype(np.float64) ** 2) / len(np.concatenate(initial_chunks))

            while True:
                chunk = sd.rec(chunk_size, samplerate=SAMPLE_RATE_HERTZ, channels=1, dtype=np.int16, blocking=True)
                chunk = np.squeeze(chunk)
                short_term_energy = np.sum(chunk.astype(np.float64) ** 2) / len(chunk)

                if short_term_energy > long_term_energy * energy_ratio_threshold:
                    if not recording:
                        recording = True
                        print("Speech detected, start recording...")
                        start_time = time.time()
                    audio_data.append(chunk)
                    silence_counter = 0
                elif recording:
                    silence_counter += chunk_duration
                    audio_data.append(chunk)
                    if silence_counter >= silence_duration:
                        print("Silence detected, stop recording...")
                        break
                if time.time() - start_time >= max_duration:
                    print("Max duration reached, stop recording...")
                    break
            
            audio_data = np.concatenate(audio_data, axis=0)
            audio_file = "recorded_audio.wav"
            sf.write(audio_file, audio_data, SAMPLE_RATE_HERTZ)

            return audio_file  # Return the path to the recorded audio file
        except Exception as e:
            logger.error(f"Error in record_audio: {e}")
            raise

    def transcribe_audio(self, audio_file):
        try:
            with open(audio_file, "rb") as file:
                content = file.read()

            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=SAMPLE_RATE_HERTZ,
                language_code=LANGUAGE_CODE,
            )

            response = self.client.recognize(config=config, audio=audio)
            return response.results[0].alternatives[0].transcript
        except Exception as e:
            logger.error(f"Error in transcribe_audio: {e}")
            raise

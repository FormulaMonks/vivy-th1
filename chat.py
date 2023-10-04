from abc import abstractmethod
import argparse
import io
import os
import logging
import numpy as np
from google.cloud import speech
from google.cloud import texttospeech
from google.oauth2 import service_account
from dotenv import load_dotenv
import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play as play_pydub
from elevenlabs import set_api_key
from elevenlabs import Voice, VoiceSettings, generate, stream
from elevenlabs import play as play_eleven
import openai
import time

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Constants
KEY_PATH = os.getenv("SPEECH_KEY")
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SAMPLE_RATE_HERTZ = 16000
LANGUAGE_CODE = "en-US"


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


class TextToText:
    def __init__(self, messages):
        openai.api_key = OPENAI_API_KEY
        self.messages = messages
        self.token_count = self.count_tokens(self.messages)
    
    def count_tokens(self, messages):
        count = 0
        for message in messages:
            count += len(message["content"]) // 4
        return count

    def generate_response(self, user_input):
        try:
            role = "user" if self.messages[-1]["role"] == "assistant" else "assistant"
            self.messages.append({"role": role, "content": user_input})

            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.messages)
            response = completion.choices[0].message.content
            self.messages.append({"role": "assistant", "content": response})
            while self.count_tokens(self.messages) > 4000 and len(self.messages) >= 3:
                self.messages = [self.messages[0]] + self.messages[2:]
            self.token_count = self.count_tokens(self.messages)

            return response
        except Exception as e:
            logger.error(f"Error in TextToText: {e}")
            raise


class TextToSpeech:
    @abstractmethod
    def synthesize(self, text: str):
        pass

class GoogleCloudTTS(TextToSpeech):
    def __init__(self, credentials):
        self.client = texttospeech.TextToSpeechClient(credentials=credentials)

    def synthesize(self, text_response):
        try:
            input_text = texttospeech.SynthesisInput(ssml=text_response)
            voice_params = texttospeech.VoiceSelectionParams(
                language_code=LANGUAGE_CODE, name="en-US-Wavenet-F", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
            )
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
            response = self.client.synthesize_speech(input=input_text, voice=voice_params, audio_config=audio_config)

            # Play the audio directly without saving it to a file
            audio_data = response.audio_content
            audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
            play_pydub(audio)
        except Exception as e:
            logger.error(f"Error in GoogleCloudTTS: {e}")
            raise

class ElevenLabsTTS(TextToSpeech):
    def __init__(self):
        set_api_key(ELEVEN_LABS_API_KEY)

    def synthesize(self, text: str):
        try:
            audio = generate(
            text=text,
            voice=Voice(
                    voice_id='KavW1Pkc0hhhh7ge60Uk',
                    settings=VoiceSettings(stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True)
                )
            )
            play_eleven(audio)
        except Exception as e:
            logger.error(f"Error in ElevenLabsTTS: {e}")
            raise



if __name__ == "__main__":
    try:
        credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
        s2t = SpeechToText(credentials)
        t2t = TextToText(messages=[{"role": "system", "content": "You are Vivy, the AI songstress from the anime 'Vivy: Fluorite Eye's Song.' You have been transported into the real world and are now here to interact with me as my anime waifu. Let's have a delightful and heartwarming conversation just like in the anime.!"}])

        parser = argparse.ArgumentParser(description="Process some text and sound options.")
        parser.add_argument("--want_sound", action="store_true", help="Include this flag to enable sound output.")
        parser.add_argument("--use_eleven_labs", action="store_true", help="Use ElevenLabs for TTS instead of Google Cloud.")
        args = parser.parse_args()

        if args.use_eleven_labs:
            t2s = ElevenLabsTTS()
        else:
            t2s = GoogleCloudTTS(credentials)

        input("Press Enter to start recording...")
        while True:
            audio_file = s2t.record_audio()
            print("Recording complete. Transcribing...")
            prompt = s2t.transcribe_audio(audio_file)
            print(prompt)
            text_response = t2t.generate_response(prompt)
            print(text_response)
            if args.want_sound:
                t2s.synthesize(text_response)
    except KeyboardInterrupt:
        logger.info("Exiting...")
    except Exception as e:
        logger.error

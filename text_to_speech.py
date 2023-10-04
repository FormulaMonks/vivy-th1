from abc import abstractmethod
import io
from google.cloud import texttospeech
from elevenlabs import set_api_key, Voice, VoiceSettings, generate, play as play_eleven
from pydub import AudioSegment
from pydub.playback import play as play_pydub
import logging
from constants import LANGUAGE_CODE, ELEVEN_LABS_API_KEY
import threading

logger = logging.getLogger(__name__)

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

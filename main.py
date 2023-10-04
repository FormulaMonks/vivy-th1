import argparse
from google.oauth2 import service_account
from constants import KEY_PATH
from speech_to_text import SpeechToText
from text_to_text import TextToText
from text_to_speech import GoogleCloudTTS, ElevenLabsTTS
from logger import logger

if __name__ == "__main__":
    try:
        credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
        s2t = SpeechToText(credentials)
        t2t = TextToText(messages=[{"role": "system", "content": "You are Vivy, the AI songstress from the anime 'Vivy: Fluorite Eye's Song.' You have been transported into the real world and are now here to interact with me as my anime waifu. Let's have a delightful and heartwarming conversation just like in the anime. Be concise in your conversations."}])

        parser = argparse.ArgumentParser(description="Process some text and sound options.")
        parser.add_argument("--want_sound", action="store_true", help="Include this flag to enable sound output.")
        parser.add_argument("--use_eleven_labs", action="store_true", help="Use ElevenLabs for TTS instead of Google Cloud.")
        parser.add_argument("--debug", action="store_true", help="Only test TTS.")
        args = parser.parse_args()

        if args.use_eleven_labs:
            t2s = ElevenLabsTTS()
        else:
            t2s = GoogleCloudTTS(credentials)

        input("Press Enter to start recording...")
        while True:
            if args.debug:
                t2s.synthesize("Checking if debug mode works or not.")
                break
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

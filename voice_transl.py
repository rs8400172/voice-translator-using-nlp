import googletrans
import speech_recognition as sr
import gtts
import spacy
from langdetect import detect, DetectorFactory
from textblob import TextBlob
from time import sleep
import random
import re
import pygame

# langdetect to a high reliability mode
DetectorFactory.seed = 0

# Load spaCy model for named entity recognition and other NLP tasks
nlp = spacy.load("en_core_web_sm")

def recognize_speech_from_mic(recognizer, microphone, retries=3):
    """Transcribe speech recorded from `microphone` with retry logic."""
    for i in range(retries):
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = recognizer.listen(source)

        try:
            transcription = recognizer.recognize_google(audio)
            return {"success": True, "error": None, "transcription": transcription}
        except sr.RequestError:
            return {"success": False, "error": "API unavailable"}
        except sr.UnknownValueError:
            return {"success": False, "error": "Unable to recognize speech"}
    return {"success": False, "error": "Max retries exceeded"}

def translate_text(text, src_language, dest_language="kn", retries=3):
    """Translate text to the specified language with retry logic."""
    translator = googletrans.Translator()
    for i in range(retries):
        try:
            translation = translator.translate(text, src=src_language, dest=dest_language)
            return {"success": True, "error": None, "translation": translation.text}
        except Exception as e:
            print(f"An error occurred during translation: {e}")
            sleep((2 ** i) + random.uniform(0, 1))  # Exponential backoff with jitter
    return {"success": False, "error": "Max retries exceeded"}

def preprocess_text(text):
    """Preprocess text by cleaning and normalizing."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def analyze_entities(text):
    """Analyze named entities in the text using spaCy."""
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def analyze_sentiment(text):
    """Analyze the sentiment of the text using TextBlob."""
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment

def detect_language(text):
    """Detect the language of the text using langdetect."""
    try:
        detected_language = detect(text)
    except Exception as e:
        print(f"Error detecting language: {e}")
        detected_language = None
    return detected_language

def play_audio(file_path):
    """Play an audio file using pygame."""
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def main():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    print("Speak Now")
    response = recognize_speech_from_mic(recognizer, microphone)

    if response["error"]:
        print(f"Error: {response['error']}")
        return

    listen = response["transcription"]
    print(f"You said: {listen}")

    # Preprocess the spoken text
    preprocessed_text = preprocess_text(listen)

    # Detect the language of the spoken text
    detected_language = detect_language(preprocessed_text)
    print(f"Detected Language: {detected_language}")

    if detected_language and detected_language != 'kn':
        translation_response = translate_text(preprocessed_text, src_language=detected_language, dest_language='kn')

        if translation_response["error"]:
            print(f"Error: {translation_response['error']}")
            return

        translated_text = translation_response["translation"]
        print(f"Translation in Kannada: {translated_text}")

        # Convert the translated text to speech
        try:
            converted_audio = gtts.gTTS(translated_text, lang="kn")
            audio_file = "translation.mp3"
            converted_audio.save(audio_file)

            # Play the converted audio
            play_audio(audio_file)
        except Exception as e:
            print(f"An error occurred during text-to-speech or playback: {e}")
    else:
        print("Either the detected language is already Kannada or language detection failed.")

    # Analyze named entities in the spoken text
    entities = analyze_entities(preprocessed_text)
    print("Named Entities:")
    for entity, label in entities:
        print(f"{entity}: {label}")

    # Analyze the sentiment of the spoken text
    sentiment = analyze_sentiment(preprocessed_text)
    print(f"Sentiment: Polarity={sentiment.polarity}, Subjectivity={sentiment.subjectivity}")

if __name__ == "__main__":
    main()

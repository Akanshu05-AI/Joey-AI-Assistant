# Joey: Your Smart In-Car AI Assistant (OkDriver Project)
# Version: 4.1 (More Content & Driver Features Enhanced)
# This version includes expanded jokes, riddles, and facts in both languages,
# driver-specific commands, and intelligent intent recognition.

import json
import os
import queue
import sounddevice as sd
import sys
import threading
import time
from datetime import datetime
from gtts import gTTS
from pygame import mixer
import pyttsx3
import random
import numpy as np
from vosk import Model, KaldiRecognizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration & Setup ---
# IMPORTANT: Paths to Vosk models
MODEL_EN_PATH = r"C:\Desktop\Projects\python projects\ok driver\new_stt\vosk-model-en-in-0.5"
MODEL_HI_PATH = r"C:\Desktop\Projects\python projects\ok driver\new_stt\vosk-model-hi-0.22"
CONFIDENCE_THRESHOLD = 0.25 # Adjusted threshold for improved recognition, especially in Hindi

# --- Main Joey Class ---

class Joey:
    """
    The core class for our voice assistant, Joey.
    Manages state, language, intelligent intent recognition, and responses.
    """
    def __init__(self):
        # --- Model and Library Checks ---
        if not os.path.exists(MODEL_EN_PATH) or not os.path.exists(MODEL_HI_PATH):
            print("\n[JOEY ERROR] Vosk models not found! Please ensure folders are in the right place.")
            print(f"Expected English model at: {MODEL_EN_PATH}")
            print(f"Expected Hindi model at: {MODEL_HI_PATH}")
            sys.exit(1)
            
        print("[JOEY] Waking up... Calibrating language modules.")
        self.model_en = Model(MODEL_EN_PATH)
        self.model_hi = Model(MODEL_HI_PATH)

        # --- State Management ---
        self.active_language = 'en'
        self.user_name = None
        self.is_listening = True
        self.context = {} # For multi-turn conversations

        # --- Text-to-Speech (TTS) Setup ---
        self.tts_engine_en = pyttsx3.init()
        self.tts_engine_en.setProperty('rate', 160)
        mixer.init() # For playing Hindi audio with gTTS

        # --- Voice Recognition Setup ---
        self.q = queue.Queue()
        self.device_info = sd.query_devices(sd.default.device[0], 'input')
        self.samplerate = int(self.device_info['default_samplerate'])
        
        # --- MASSIVELY EXPANDED Intent Dictionary for Demo ---
        self.intents = {
            # --- Conversational ---
            'greeting': ['hello', 'hi', 'hey joey', 'namaste', 'ok driver', 'hey', 'start', 'wake up joey', 'joey', 'नमस्ते'],
            'small_talk_how_are_you': [
                'how are you', 'what\'s up', 'kya hal hai', 'kya chal raha hai', # Keep Romanized if Vosk gives Romanized for these
                'कैसे हो', 'आप कैसे हो', 'आप कैसे हैं', # CHANGE TO DEVANAGARI
                'कैसा है', 'तुम कैसे हो', 'आप ठीक हैं', 'आपकी तबियत कैसी है' # CHANGE TO DEVANAGARI
            ],
            'introduce_self': ['my name is', 'i am', 'मेरा नाम है', 'मैं हूँ'], # CHANGE TO DEVANAGARI
            'thank_you': ['thank you', 'thanks a lot', 'शुक्रिया', 'धन्यवाद', 'बहुत शुक्रिया', 'thank u', 'शुक्रीया'], # CHANGE TO DEVANAGARI
            'goodbye': ['goodbye', 'bye', 'exit', 'shut down', 'बंद हो जाओ', 'अलविदा', 'stop', 'टाटा'], # CHANGE TO DEVANAGARI
            
            # --- Language Switching ---
            'change_language_hi': [
                'hindi mode', 'switch to hindi', 'hindi mein bolo', 'hindi mein baat karo', 'talk to me in hindi',
                'change to hindi', 'hindi chalu karo', 'hindi mein baat kar', 'hindi bhasha', 'हिंदी में बोलो',
                'हिंदी मोड', 'हिंदी चलाओ' # CHANGED TO DEVANAGARI
            ],
            'change_language_en': [
                'english mode', 'switch to english', 'english mein bolo', 'speak in english', 'talk to me in english',
                'change to english', 'english chalu karo', 'english mein baat kar', 'english bhasha',
                'अंग्रेजी में बात करें', 'इंग्लिश में बोलो', 'वापस इंग्लिश में', 'इंग्लिश मोड में आओ' # CHANGED TO DEVANAGARI
            ],

            # --- Core Features ---
            'ask_name': ["what's your name", 'what is your name', 'who are you', 'तुम्हारा नाम क्या है', 'आपका नाम क्या है', 'तुम कौन हो', 'what do people call you'], # CHANGE TO DEVANAGARI
            'ask_creator': ['who made you', 'who created you', 'तुम्हें किसने बनाया', 'आपको किसने बनाया', 'who is your developer'], # CHANGE TO DEVANAGARI
            'ask_time': ['what time is it', 'time kya hua hai', 'समय क्या हुआ है', 'tell me the time', 'टाइम बताओ', 'घड़ी में क्या बजा है', 'कितने बजे हैं'], # CHANGE TO DEVANAGARI
            'ask_weather': [
                "what's the weather", 'how is the weather', 'weather kaisa hai', 'मौसम का हाल', 'आज का मौसम कैसा है',
                'weather update do', 'मौसम की जानकारी दो', 'आज का वेदर कैसा था', 'आज का वेदर कैसा था', 'will it rain today',
                'कल का मौसम', 'आज का तापमान', 'बारिश होगी' # CHANGE TO DEVANAGARI
            ],
            'ask_location': ['where are we', 'current location', 'हम कहाँ हैं', 'मेरी लोकेशन बताओ', 'हमारी लोकेशन क्या है', 'where am i', 'लोकेशन बताओ', 'मेरी स्थिति क्या है'], # CHANGE TO DEVANAGARI
            'vehicle_status': ['car status', 'vehicle status', 'check the car', 'गाड़ी की स्थिति', 'मेरी गाड़ी का स्टेटस बताओ', 'what\'s my car\'s status', 'इंजन चेक', 'गाड़ी का हाल'], # CHANGE TO DEVANAGARI
            
            # --- NEW DRIVER FEATURES ---
            'ask_fuel_level': ['what\'s the fuel level', 'how much fuel is left', 'fuel status', 'check fuel', 'कितना पेट्रोल बचा है', 'फ्यूल कितना है', 'तेल कितना है', 'गाड़ी में पेट्रोल कितना है', 'ईंधन का स्तर बताओ'],
            'ask_tire_pressure': ['what\'s the tire pressure', 'check tire pressure', 'tire pressure status', 'टायर में हवा कितनी है', 'टायर प्रेशर बताओ', 'टायर का दबाव कैसा है'],
            'increase_temperature': ['increase temperature', 'make it warmer', 'temperature up', 'turn up the heat', 'तापमान बढ़ाओ', 'गर्मी बढ़ाओ', 'टेंपरेचर बढ़ाओ'],
            'decrease_temperature': ['decrease temperature', 'make it cooler', 'temperature down', 'turn down the heat', 'तापमान घटाओ', 'ठंडा करो', 'टेंपरेचर कम करो'],
            'turn_ac_on': ['turn ac on', 'switch on ac', 'ac chalao', 'एसी चलाओ', 'एसी ऑन करो', 'एसी चालू करो'],
            'turn_ac_off': ['turn ac off', 'switch off ac', 'ac band karo', 'एसी बंद करो', 'एसी ऑफ़ करो'],
            'find_nearest': ['find nearest', 'where is the nearest', 'locate nearest', 'सबसे नज़दीकी कहाँ है', 'नज़दीकी ढूंढो'], # Entity extracted for what to find
            'traffic_update': ['traffic update', 'how\'s the traffic', 'is there traffic ahead', 'ट्रैफिक बताओ', 'यातायात कैसा है', 'आगे ट्रैफिक है क्या'],
            'ask_eta': ['what\'s my eta', 'estimated time of arrival', 'when will i arrive', 'पहुँचने में कितना समय लगेगा', 'ईटीए बताओ', 'कब पहुंचेंगे'],
            'headlights_on': ['turn headlights on', 'headlights on', 'हेडलाइट ऑन करो', 'लाइट जलाओ'],
            'headlights_off': ['turn headlights off', 'headlights off', 'हेडलाइट बंद करो', 'लाइट बुझाओ'],
            # --- END NEW DRIVER FEATURES ---

            'navigate': [
                'navigate to', 'take me to', 'mujhe le jao', 'get directions to', 'mujhe jana hai', 'chalo', 'go to', 'drive to',
                'मुझे घर ले चलो', 'मुझे हॉस्पिटल ले जाओ', 'नेविगेशन शुरू करो', 'रास्ता बताओ', 'नेविगेट करो' # CHANGE TO DEVANAGARI
            ],
            'cancel_navigation': ['cancel navigation', 'stop navigation', 'रुक जाओ', 'नेविगेशन बंद करो', 'रूट कैंसिल करो'], # CHANGE TO DEVANAGARI
            
            # --- Entertainment ---
            'tell_joke': ['tell me a joke', 'जोक सुनाओ', 'कोई चुटकुला सुनाओ', 'make me laugh', 'हंसाओ', 'एक जोक सुनाओ'], # CHANGE TO DEVANAGARI
            'joke_feedback': ['not funny', 'that was a bad joke', 'बेकार था', 'अच्छा नहीं था', 'lame joke', 'मजा नहीं आया'], # CHANGE TO DEVANAGARI
            'tell_riddle': ['tell me a riddle', 'ask me a riddle', 'रिडल पूछो', 'एक पहेली पूछो', 'पहेली सुनाओ'], # CHANGE TO DEVANAGARI
            'answer_riddle': ['what is the answer', 'आंसर क्या है', 'tell me the solution', 'जवाब बताओ', 'रिडल का आंसर'], # CHANGE TO DEVANAGARI
            'tell_fact': ['tell me a fact', 'share a fact', 'कोई तथ्य बताओ', 'ज्ञान की बात', 'इंट्रेस्टिंग फैक्ट'], # CHANGE TO DEVANAGARI

            # --- Music Control (Simulated) ---
            'play_music': ['play music', 'play a song', 'गाना बजाओ', 'म्यूजिक चलाओ', 'play something', 'सॉन्ग चलाओ'], # CHANGE TO DEVANAGARI
            'pause_music': ['pause music', 'stop music', 'गाना रोको', 'म्यूजिक बंद करो', 'पॉज सॉन्ग'], # CHANGE TO DEVANAGARI
            'next_song': ['next song', 'अगला गाना', 'skip song', 'नेक्स्ट ट्रैक'], # CHANGE TO DEVANAGARI
            'previous_song': ['previous song', 'पिछला गाना', 'प्रीवियस ट्रैक'], # CHANGE TO DEVANAGARI
            'volume_up': ['volume up', 'आवाज बढ़ाओ', 'raise volume', 'वॉल्यूम तेज करो'], # CHANGE TO DEVANAGARI
            'volume_down': ['volume down', 'आवाज कम करो', 'lower volume', 'वॉल्यूम धीमा करो'], # CHANGE TO DEVANAGARI

            # --- Calling/Messaging (Simulated) ---
            'make_call': ['call', 'make a call', 'फ़ोन लगा दो', 'call someone', 'किसी को कॉल करो'], # CHANGE TO DEVANAGARI
            'send_message': ['send message', 'text', 'मैसेज भेजो', 'टेक्स्ट करो'], # CHANGE TO DEVANAGARI
            
            # --- Utility & Help ---
            'translate': ['translate', 'can you translate', 'ट्रांसलेट कर दो', 'अनुवाद करो'], # CHANGE TO DEVANAGARI
            'help': ['what can you do', 'help', 'help me', 'क्या कर सकते हो', 'मदद करो', 'capabilities', 'commands', 'व्हाट कैन आई आस्क'], # CHANGE TO DEVANAGARI
            'emergency': ['emergency', 'call police', 'call ambulance', 'i need help', 'trouble', 'sos', 'हेल्प मी नाउ'], # CHANGE TO DEVANAGARI
            'set_reminder': ['set a reminder', 'remind me', 'मुझे याद दिलाओ', 'रिमाइंडर सेट करो'], # CHANGE TO DEVANAGARI
            'open_app': ['open app', 'launch app', 'ऐप खोलो'], # CHANGE TO DEVANAGARI
            'what_is': ['what is', 'define', 'क्या है'], # CHANGE TO DEVANAGARI
        }
        
        # --- TF-IDF Setup for Intent Recognition ---
        self.corpus = []
        self.intent_map = []
        for intent, phrases in self.intents.items():
            for phrase in phrases:
                self.corpus.append(phrase)
                self.intent_map.append(intent)

        self.vectorizer = TfidfVectorizer()
        self.corpus_vectors = self.vectorizer.fit_transform(self.corpus)
        print("[JOEY] Language modules calibrated. Ready for your command!")

        # --- Response Data ---
        self.jokes = {
            'en': [
                "Why don't scientists trust atoms? Because they make up everything!",
                "I told my wife she should embrace her mistakes. She gave me a hug.",
                "What do you call a fake noodle? An Impasta!",
                "Why did the scarecrow win an award? Because he was outstanding in his field!",
                "What do you call a pile of cats? A meowtain!",
                "I'm reading a book about anti-gravity. It's impossible to put down!"
            ],
            'hi': [
                "टीचर: 'बच्चों, मुश्किल का सामना हिम्मत से करना चाहिए।' पप्पू: 'सर, कल रात मैंने मुश्किल का सामना हिम्मत से किया, पर मुश्किल के पापा ने मुझे बहुत मारा!'",
                "सांता: 'यार, मैं अपनी कार की चाबी ढूंढ रहा हूं।' बंता: 'अरे, कार में ही चेक कर लो।' सांता: 'नहीं कर सकता, कार तो लॉक है!'",
                "टीचर: 'तुम्हारा सबसे अच्छा दोस्त कौन है?' छात्र: 'मेरी किताब।' टीचर: 'अच्छा, यह बताओ कि अगर मैं तुम्हारी किताब छीन लूं, तो?' छात्र: 'तो आप मेरे सबसे अच्छे दोस्त बन जाएंगे!'",
                "बेटा: 'मां, मैं कल स्कूल नहीं जाऊंगा।' मां: 'क्यों?' बेटा: 'मेरे दोस्त मुझे पसंद नहीं करते और टीचर मुझे समझती नहीं।' मां: 'नहीं बेटा, तुम स्कूल जरूर जाओगे, क्योंकि तुम स्कूल के प्रिंसिपल हो!'",
                "मरीज: 'डॉक्टर साहब, मुझे नींद नहीं आती।' डॉक्टर: 'तो सोने की कोशिश क्यों नहीं करते?' मरीज: 'वही तो कर रहा हूं, लेकिन फिर नींद नहीं आती!'"
            ]
        }
        self.riddles = {
            'en': {
                "riddle": "I have cities, but no houses. I have mountains, but no trees. I have water, but no fish. What am I?", "answer": "A map"
            },
            'hi': {
                "riddle": "ऐसी कौन सी चीज है जो पानी पीते ही मर जाती है?", "answer": "प्यास (Thirst)"
            },
            # Additional riddles
            'en_extra': [
                {"riddle": "I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I?", "answer": "An echo"},
                {"riddle": "What has to be broken before you can use it?", "answer": "An egg"},
                {"riddle": "What is full of holes but still holds water?", "answer": "A sponge"}
            ],
            'hi_extra': [
                {"riddle": "वह कौन सी चीज़ है जो जितनी ज्यादा बढ़ती है, उतनी ही कम होती जाती है?", "answer": "उम्र (Age)"},
                {"riddle": "जितनी ज्यादा तुम मुझे निकालोगे, उतनी ही मैं बड़ी होती जाऊंगी। मैं क्या हूँ?", "answer": "एक गड्ढा (A hole)"},
                {"riddle": "कटने पर तुम रोते हो, मुझे बिना खाए कोई नहीं रह सकता। मैं क्या हूँ?", "answer": "प्याज (Onion)"}
            ]
        }
        self.facts = {
            'en': [
                "A single cloud can weigh more than a million pounds.",
                "The unicorn is the national animal of Scotland.",
                "Bananas are berries, but strawberries aren't.",
                "Honey never spoils.",
                "A group of owls is called a parliament.",
                "The shortest war in history lasted only 38 to 45 minutes. (Between Britain and Zanzibar in 1896)",
                "The average person walks the equivalent of three times around the world in their lifetime.",
                "Bees can fly higher than Mount Everest."
            ],
            'hi': [
                "ऊंट (camel) के दूध का दही नहीं जमता।",
                "अंटार्कटिका में चींटियां (ants) नहीं पाई जाती हैं।",
                "इंसान के शरीर की सबसे छोटी हड्डी कान में होती है।",
                "बिल्लियां (cats) 1000 से ज्यादा अलग-अलग आवाजें निकाल सकती हैं।",
                "पृथ्वी का सबसे गहरा बिंदु प्रशांत महासागर में मारियाना ट्रेंच (Mariana Trench) है।",
                "इंसान का दिल एक दिन में लगभग 100,000 बार धड़कता है।",
                "नीली व्हेल (blue whale) पृथ्वी पर सबसे बड़ा जानवर है।",
                "शहद कभी खराब नहीं होता। पुरातत्वविदों (archaeologists) ने हज़ारों साल पुराने शहद के बर्तन खोजे हैं जो अभी भी खाने लायक थे।"
            ]
        }
        
    def speak(self, text, lang=None):
        """Handles text-to-speech, defaulting to the active language."""
        lang_to_use = lang or self.active_language
        print(f"[JOEY SPEAKS ({lang_to_use})] >> {text}")
        if lang_to_use == 'hi':
            try:
                tts = gTTS(text=text, lang='hi', slow=False)
                tts.save("response.mp3")
                mixer.music.load("response.mp3")
                mixer.music.play()
                while mixer.music.get_busy(): time.sleep(0.1)
                mixer.music.unload()
                os.remove("response.mp3")
            except Exception as e:
                print(f"[JOEY TTS ERROR] Could not play Hindi audio: {e}")
                self.speak("Sorry, I'm having a little trouble speaking Hindi.", 'en')
        else: # English
            self.tts_engine_en.say(text)
            self.tts_engine_en.runAndWait()

    def recognize_intent(self, text):
        """Uses TF-IDF for intent recognition and rule-based for entity extraction."""
        text = text.lower().strip()
        if not text: return None, None
        
        command_vector = self.vectorizer.transform([text])
        similarities = cosine_similarity(command_vector, self.corpus_vectors)
        max_score_index = np.argmax(similarities)
        confidence = similarities[0, max_score_index]

        # --- DEBUGGING LINE ---
        # This will show you the highest matched intent and its confidence score
        if self.intent_map: # Check if intent_map is populated to avoid IndexError
            print(f"DEBUG: Intent candidate: '{self.intent_map[max_score_index]}' with confidence: {confidence:.2f} for text: '{text}'")
        # --- END DEBUGGING LINE ---

        if confidence < CONFIDENCE_THRESHOLD:
            return None, None
        
        intent = self.intent_map[max_score_index]
        entity = None

        # More robust entity extraction for specific intents
        if intent == 'navigate':
            phrases_to_remove = ['navigate to', 'take me to', 'mujhe le jao', 'get directions to', 'mujhe jana hai', 'chalo', 'go to', 'drive to', 'mujhe ghar le chalo', 'mujhe hospital le jaao', 'navigasyon shuru karo', 'rasta batao', 'navigated to', 'मुझे ले जाओ', 'मुझे जाना है', 'चलो', 'मुझे घर ले चलो', 'मुझे अस्पताल ले जाओ', 'नेविगेशन शुरू करो', 'रास्ता बताओ', 'नेविगेट करो']
            for phrase in phrases_to_remove:
                if text.startswith(phrase):
                    entity = text.replace(phrase, '', 1).strip()
                    if entity: break
        elif intent == 'introduce_self':
            phrases_to_remove = ['my name is', 'i am', 'मेरा नाम है', 'मैं हूँ']
            for phrase in phrases_to_remove:
                if phrase in text:
                    entity = text.split(phrase, 1)[-1].strip()
                    if entity: break
        elif intent == 'make_call':
            if 'call' in text or 'फ़ोन लगा दो' in text or 'किसी को कॉल करो' in text:
                entity = text.replace('call', '', 1).replace('फ़ोन लगा दो', '', 1).replace('किसी को कॉल करो', '', 1).strip()
        elif intent == 'send_message':
            if 'send message to' in text or 'मैसेज भेजो' in text:
                entity = text.replace('send message to', '', 1).replace('मैसेज भेजो', '', 1).strip()
            elif 'text' in text:
                entity = text.replace('text', '', 1).strip()
        elif intent == 'set_reminder':
            if 'remind me to' in text or 'मुझे याद दिलाओ' in text:
                entity = text.replace('remind me to', '', 1).replace('मुझे याद दिलाओ', '', 1).strip()
            elif 'set a reminder for' in text:
                entity = text.replace('set a reminder for', '', 1).strip()
        elif intent == 'open_app':
            if 'open' in text or 'ऐप खोलो' in text:
                entity = text.replace('open', '', 1).replace('ऐप खोलो', '', 1).strip()
            elif 'launch' in text:
                entity = text.replace('launch', '', 1).strip()
        elif intent == 'what_is':
            if 'what is' in text or 'क्या है' in text:
                entity = text.replace('what is', '', 1).replace('क्या है', '', 1).strip()
            elif 'define' in text:
                entity = text.replace('define', '', 1).strip()
        # --- NEW ENTITY EXTRACTION FOR DRIVER FEATURES ---
        elif intent == 'find_nearest':
            phrases_to_remove = ['find nearest', 'where is the nearest', 'locate nearest', 'सबसे नज़दीकी कहाँ है', 'नज़दीकी ढूंढो']
            for phrase in phrases_to_remove:
                if text.startswith(phrase):
                    entity = text.replace(phrase, '', 1).strip()
                    if entity: break
        # --- END NEW ENTITY EXTRACTION ---
        
        return intent, entity
        
    def _audio_callback(self, indata, frames, time, status):
        """Captures audio data into a queue."""
        if status: print(status, file=sys.stderr)
        self.q.put(bytes(indata))

    def listen(self):
        """
        Smarter listening: Uses both models but prioritizes the one matching the active language.
        If active language model yields no result, it checks the other model.
        """
        print("\n[JOEY] Listening...")
        rec_en = KaldiRecognizer(self.model_en, self.samplerate)
        rec_hi = KaldiRecognizer(self.model_hi, self.samplerate)

        # Resetting recognizers for a clean start to avoid residual data
        rec_en.SetWords(True) # Ensure words are recognized
        rec_hi.SetWords(True)

        with sd.RawInputStream(samplerate=self.samplerate, blocksize=8000, device=self.device_info['index'],
                               dtype='int16', channels=1, callback=self._audio_callback):
            
            start_time = time.time()
            # Listen for a maximum of 5 seconds to prevent hanging
            while (time.time() - start_time) < 5: 
                data = self.q.get(timeout=1.0) # Get with a timeout to prevent infinite wait
                
                # Try with active language model first
                if self.active_language == 'en':
                    if rec_en.AcceptWaveform(data):
                        text_en = json.loads(rec_en.Result()).get('text', '')
                        if text_en:
                            print(f"[VOSK] Heard (EN primary): '{text_en}'")
                            return text_en
                    rec_hi.AcceptWaveform(data) # Still process Hindi in background
                else: # self.active_language == 'hi'
                    if rec_hi.AcceptWaveform(data):
                        text_hi = json.loads(rec_hi.Result()).get('text', '')
                        if text_hi:
                            print(f"[VOSK] Heard (HI primary): '{text_hi}'")
                            return text_hi
                    rec_en.AcceptWaveform(data) # Still process English in background
            
            # If nothing definitive was heard, try the final results of both
            final_en_res = json.loads(rec_en.FinalResult()).get('text', '')
            final_hi_res = json.loads(rec_hi.FinalResult()).get('text', '')

            # Choose the longest and most likely non-empty result
            if final_en_res and (not final_hi_res or len(final_en_res) >= len(final_hi_res)):
                chosen_text = final_en_res
            elif final_hi_res:
                chosen_text = final_hi_res
            else:
                chosen_text = "" # Nothing heard

            if chosen_text:
                print(f"[VOSK] Final Heard: '{chosen_text}' (en: '{final_en_res}', hi: '{final_hi_res}')")
            else:
                print("[VOSK] No clear command heard.")
            return chosen_text


    def handle_command(self, text):
        """Processes the recognized text and triggers the appropriate action, including multi-turn context."""
        
        # --- Context Handling ---
        # Riddle context
        if self.context.get('state') == 'awaiting_riddle_answer':
            if 'answer' in text.lower() or 'jawab' in text.lower() or 'solution' in text.lower() or 'batao' in text.lower() or 'जवाब बताओ' in text or 'आंसर क्या है' in text:
                answer = self.context.get('riddle_answer', "I forgot the answer myself!")
                self.speak(f"The answer is... {answer}", self.active_language) # Specify language
                self.context = {} # Clear context after answer
            else:
                self.speak("That's an interesting guess! But if you want the answer, just say 'what is the answer?'", self.active_language)
            return # Exit after handling context

        # Translation context (multi-turn)
        if self.context.get('state') == 'awaiting_translation_phrase':
            self.context['phrase_to_translate'] = text
            self.context['state'] = 'awaiting_target_language'
            self.speak("Got it. And should I translate that to English or Hindi?", self.active_language)
            return

        if self.context.get('state') == 'awaiting_target_language':
            phrase = self.context.get('phrase_to_translate', 'that')
            target_lang_spoken = 'en' if 'english' in text.lower() or 'इंग्लिश' in text else 'hi' # Crude but effective for demo
            
            # Simulated translation response
            if target_lang_spoken == 'hi':
                self.speak(f"Simulated translation: '{phrase}' in Hindi would be 'यह एक डेमो अनुवाद है'.", self.active_language)
            else:
                self.speak(f"Simulated translation: '{phrase}' in English would be 'This is a demo translation'.", self.active_language)
            self.context = {} # Clear context
            return


        # --- Intent Recognition and Handling ---
        intent, entity = self.recognize_intent(text)
        
        if intent == 'greeting':
            name_part = f", {self.user_name}" if self.user_name else ""
            responses = {'en': f"Hello{name_part}! How can I help you today?", 'hi': f"नमस्ते{name_part}! मैं आपकी क्या सहायता कर सकती हूँ?"}
            self.speak(responses[self.active_language])

        elif intent == 'small_talk_how_are_you':
            responses = {'en': "I'm doing great, thanks for asking! Ready for the road.", 'hi': "मैं ठीक हूँ, पूछने के लिए शुक्रिया! सफ़र के लिए तैयार।"}
            self.speak(responses[self.active_language])

        elif intent == 'introduce_self' and entity:
            self.user_name = entity.split()[0].capitalize() # Take the first word as the name
            responses = {'en': f"Got it! Nice to meet you, {self.user_name}.", 'hi': f"समझ गई! आपसे मिलकर खुशी हुई, {self.user_name}."}
            self.speak(responses[self.active_language])

        elif intent == 'change_language_hi':
            if self.active_language == 'hi':
                self.speak("मैं पहले से ही हिंदी में बात कर रही हूँ।", 'hi')
            else:
                self.active_language = 'hi'
                self.speak("ठीक है, अब मैं हिंदी में बात करूंगी।", 'hi')

        elif intent == 'change_language_en':
            if self.active_language == 'en':
                self.speak("I'm already speaking in English.", 'en')
            else:
                self.active_language = 'en'
                self.speak("Alright, switching back to English.", 'en')
        
        elif intent == 'ask_name':
            responses = {'en': "You can call me Joey. I'm your friendly co-pilot.", 'hi': "आप मुझे Joey बुला सकते हैं। मैं सफ़र में आपकी दोस्त हूँ।"}
            self.speak(responses[self.active_language])

        elif intent == 'ask_creator':
            self.speak("I was brought to life by a team of clever developers for the OkDriver project. It's nice to be here!", self.active_language)

        elif intent == 'ask_time':
            current_time = datetime.now().strftime("%I:%M %p")
            responses = {'en': f"It's {current_time}.", 'hi': f"अभी समय है {current_time} बजे।"}
            self.speak(responses[self.active_language])

        elif intent == 'ask_weather':
            responses = {'en': "Simulated weather for Faridabad is currently pleasant, around 28 degrees Celsius with clear skies.", 'hi': "फ़रीदाबाद में मौसम सुहाना है, लगभग 28 डिग्री सेल्सियस, और आसमान साफ़ है।"}
            self.speak(responses[self.active_language])

        elif intent == 'ask_location':
            responses = {'en': "Based on my simulated GPS, we are currently in Faridabad, Haryana, India.", 'hi': "मेरे सिमुलेटेड GPS के अनुसार, हम अभी फ़रीदाबाद, हरियाणा, भारत में हैं।"}
            self.speak(responses[self.active_language])

        elif intent == 'vehicle_status':
            responses = {'en': "Simulated status: Everything looks good! Fuel is at 80%, tire pressure is normal, and we're cruising at a safe speed.", 'hi': "सिमुलेटेड स्टेटस: सब ठीक लग रहा है! फ़्यूल 80 प्रतिशत है, टायर प्रेशर सामान्य है, और हम एक सुरक्षित गति पर चल रहे हैं।"}
            self.speak(responses[self.active_language])
            
        # --- NEW DRIVER FEATURE RESPONSES ---
        elif intent == 'ask_fuel_level':
            responses = {'en': "Simulated: Your fuel level is at 75%. Plenty for your journey.", 'hi': "सिमुलेटेड: आपका फ़्यूल लेवल 75% है। आपकी यात्रा के लिए काफ़ी है।"}
            self.speak(responses[self.active_language])

        elif intent == 'ask_tire_pressure':
            responses = {'en': "Simulated: All tire pressures are normal, around 32 PSI.", 'hi': "सिमुलेटेड: सभी टायरों का प्रेशर सामान्य है, लगभग 32 PSI।"}
            self.speak(responses[self.active_language])

        elif intent == 'increase_temperature':
            responses = {'en': "Simulated: Increasing cabin temperature slightly. It's now 22 degrees Celsius.", 'hi': "सिमुलेटेड: केबिन का तापमान थोड़ा बढ़ा रही हूँ। अब यह 22 डिग्री सेल्सियस है।"}
            self.speak(responses[self.active_language])

        elif intent == 'decrease_temperature':
            responses = {'en': "Simulated: Decreasing cabin temperature. It's now 19 degrees Celsius.", 'hi': "सिमुलेटेड: केबिन का तापमान घटा रही हूँ। अब यह 19 डिग्री सेल्सियस है।"}
            self.speak(responses[self.active_language])

        elif intent == 'turn_ac_on':
            responses = {'en': "Simulated: Turning on the air conditioning.", 'hi': "सिमुलेटेड: एयर कंडीशनिंग चालू कर रही हूँ।"}
            self.speak(responses[self.active_language])

        elif intent == 'turn_ac_off':
            responses = {'en': "Simulated: Turning off the air conditioning.", 'hi': "सिमुलेटेड: एयर कंडीशनिंग बंद कर रही हूँ।"}
            self.speak(responses[self.active_language])

        elif intent == 'find_nearest' and entity:
            responses = {
                'en': f"Simulated: Searching for the nearest {entity}. I found one 2 kilometers away.",
                'hi': f"सिमुलेटेड: सबसे नज़दीकी {entity} ढूंढ रही हूँ। मुझे 2 किलोमीटर दूर एक मिला।"
            }
            self.speak(responses[self.active_language])
        elif intent == 'find_nearest' and not entity:
            responses = {
                'en': "What are you looking for? (e.g., gas station, restaurant, hospital, parking)",
                'hi': "आप क्या ढूंढ रहे हैं? (उदाहरण के लिए, पेट्रोल पंप, रेस्टोरेंट, अस्पताल, पार्किंग)"
            }
            self.speak(responses[self.active_language])

        elif intent == 'traffic_update':
            responses = {'en': "Simulated: Current traffic is light on your route. No major delays reported.", 'hi': "सिमुलेटेड: आपके मार्ग पर वर्तमान यातायात हल्का है। कोई बड़ी देरी नहीं बताई गई है।"}
            self.speak(responses[self.active_language])

        elif intent == 'ask_eta':
            responses = {'en': "Simulated: Your estimated time of arrival is 3:30 PM.", 'hi': "सिमुलेटेड: आपके पहुंचने का अनुमानित समय दोपहर 3:30 बजे है।"}
            self.speak(responses[self.active_language])
            
        elif intent == 'headlights_on':
            responses = {'en': "Simulated: Headlights are now on.", 'hi': "सिमुलेटेड: हेडलाइट्स अब चालू हैं।"}
            self.speak(responses[self.active_language])

        elif intent == 'headlights_off':
            responses = {'en': "Simulated: Headlights are now off.", 'hi': "सिमुलेटेड: हेडलाइट्स अब बंद हैं।"}
            self.speak(responses[self.active_language])
        # --- END NEW DRIVER FEATURE RESPONSES ---

        elif intent == 'navigate' and entity:
            responses = {'en': f"Okay, starting simulated navigation to {entity}. Let's go!", 'hi': f"ठीक है, {entity} के लिए सिमुलेटेड नेविगेशन शुरू कर रही हूँ। चलिए!"}
            self.speak(responses[self.active_language])

        elif intent == 'cancel_navigation':
            responses = {'en': "Okay, canceling the current navigation.", 'hi': "ठीक है, नेविगेशन रद्द कर रही हूँ।"}
            self.speak(responses[self.active_language])

        elif intent == 'play_music':
            responses = {'en': "Simulated: Playing some relaxing tunes for your drive.", 'hi': "सिमुलेटेड: आपकी ड्राइव के लिए थोड़ा आरामदायक संगीत बजा रही हूँ।"}
            self.speak(responses[self.active_language])
        
        elif intent == 'pause_music':
            responses = {'en': "Simulated: Music paused.", 'hi': "सिमुलेटेड: संगीत रोक दिया गया है।"}
            self.speak(responses[self.active_language])

        elif intent == 'next_song':
            responses = {'en': "Simulated: Skipping to the next song.", 'hi': "सिमुलेटेड: अगले गाने पर जा रही हूँ।"}
            self.speak(responses[self.active_language])

        elif intent == 'previous_song':
            responses = {'en': "Simulated: Going back to the previous song.", 'hi': "सिमुलेटेड: पिछले गाने पर जा रही हूँ।"}
            self.speak(responses[self.active_language])

        elif intent == 'volume_up':
            responses = {'en': "Simulated: Turning volume up.", 'hi': "सिमुलेटेड: आवाज़ बढ़ा रही हूँ।"}
            self.speak(responses[self.active_language])
        
        elif intent == 'volume_down':
            responses = {'en': "Simulated: Turning volume down.", 'hi': "सिमुलेटेड: आवाज़ कम कर रही हूँ।"}
            self.speak(responses[self.active_language])

        elif intent == 'make_call' and entity:
            responses = {'en': f"Simulated: Calling {entity}. This is a demo feature.", 'hi': f"सिमुलेटेड: {entity} को कॉल कर रही हूँ। यह एक डेमो फ़ीचर है।"}
            self.speak(responses[self.active_language])
        elif intent == 'make_call' and not entity:
            responses = {'en': "Whom would you like to call?", 'hi': "किसको कॉल करना चाहते हैं?"}
            self.speak(responses[self.active_language])

        elif intent == 'send_message' and entity:
            responses = {'en': f"Simulated: Sending a message to {entity}. What's the message?", 'hi': f"सिमुलेटेड: {entity} को मैसेज भेज रही हूँ। क्या मैसेज है?"}
            self.speak(responses[self.active_language])
            # Add context for message content if needed
        elif intent == 'send_message' and not entity:
            responses = {'en': "Whom should I send the message to?", 'hi': "किसको मैसेज भेजना है?"}
            self.speak(responses[self.active_language])

        elif intent == 'set_reminder' and entity:
            responses = {'en': f"Simulated: Okay, I'll remind you to {entity}.", 'hi': f"सिमुलेटेड: ठीक है, मैं आपको {entity} के लिए याद दिलाऊंगी।"}
            self.speak(responses[self.active_language])
        elif intent == 'set_reminder' and not entity:
            responses = {'en': "What should I remind you about?", 'hi': "किस बारे में याद दिलाऊं?"}
            self.speak(responses[self.active_language])

        elif intent == 'open_app' and entity:
            responses = {'en': f"Simulated: Opening {entity}. This is a demo feature.", 'hi': f"सिमुलेटेड: {entity} खोल रही हूँ। यह एक डेमो फ़ीचर है।"}
            self.speak(responses[self.active_language])
        elif intent == 'open_app' and not entity:
            responses = {'en': "Which app would you like to open?", 'hi': "कौन सा ऐप खोलना चाहते हैं?"}
            self.speak(responses[self.active_language])

        elif intent == 'what_is' and entity:
            responses = {'en': f"Simulated: Searching for '{entity}'. For the demo, I'll say it's an important concept!", 'hi': f"सिमुलेटेड: '{entity}' की जानकारी ढूंढ रही हूँ। डेमो के लिए, मैं कहूंगी यह एक महत्वपूर्ण अवधारणा है!"}
            self.speak(responses[self.active_language])
        elif intent == 'what_is' and not entity:
            responses = {'en': "What would you like to know about?", 'hi': "किस बारे में जानना चाहते हैं?"}
            self.speak(responses[self.active_language])

        elif intent == 'help':
            responses = {'en': "I can tell you the time, weather, share jokes or facts, and simulate navigation, music control, calls, or messages. I can also help with car status, temperature, and finding nearby places. You can switch my language to Hindi. Just ask!", 'hi': "मैं आपको समय, मौसम, जोक्स या तथ्य बता सकती हूँ, और नेविगेशन, संगीत, कॉल, या मैसेज को सिमुलेट कर सकती हूँ। मैं गाड़ी की स्थिति, तापमान और आस-पास की जगहें ढूंढने में भी मदद कर सकती हूँ। आप मेरी भाषा हिंदी में भी बदल सकते हैं। बस पूछिए!"}
            self.speak(responses[self.active_language])
            
        elif intent == 'thank_you':
            responses = {'en': ["You're welcome!", "Anytime!", "Glad I could help!"], 'hi': ["कोई बात नहीं!", "आपका स्वागत है।", "खुशी हुई मदद करके!"]}
            self.speak(random.choice(responses[self.active_language]))

        elif intent == 'emergency':
            self.speak("This is a serious situation. Please contact emergency services directly. I cannot make real calls.", 'en')
        
        elif intent == 'goodbye':
            name_part = f", {self.user_name}" if self.user_name else ""
            self.speak(f"Goodbye{name_part}! Drive safe.", self.active_language)
            self.is_listening = False

        else: # If intent is None or not handled above
            responses = {
                'en': "Sorry, I didn't quite get that. Could you please rephrase?", 
                'hi': "माफ़ कीजिए, मैं समझ नहीं पाई। क्या आप दूसरे शब्दों में दोहरा सकते हैं?"
            }
            self.speak(responses[self.active_language])

    def test_microphone_recording(self):
        """Runs a diagnostic to check microphone input levels."""
        print("\n--- Microphone Diagnostic Test ---")
        self.speak("Let's test your microphone. Please say 'Hello Joey' after the beep.", 'en')
        # A simple beep sound
        sd.play(0.3 * np.sin(2 * np.pi * 440 * np.arange(44100) / 44100), samplerate=44100)
        time.sleep(1) # Give a moment for the beep to play

        try:
            myrecording = sd.rec(int(3 * self.samplerate), samplerate=self.samplerate, channels=1, dtype='int16')
            sd.wait() # Wait for the recording to finish
            
            # Simple volume check (normalize and check magnitude)
            volume_norm = np.linalg.norm(myrecording) * 10
            if volume_norm > 1000: # Threshold for reasonable sound (adjust if too sensitive/insensitive)
                self.speak("Great! I can hear you loud and clear.", 'en')
            else:
                self.speak("I couldn't hear you very well. Please check if your microphone is selected and not muted.", 'en')
        except Exception as e:
            self.speak("I ran into an error with your microphone. Please make sure it's connected and enabled.", 'en')
            print(f"[JOEY ERROR] Microphone test failed: {e}")
            return False
        return True

    def start(self):
        """The main loop of the assistant."""
        if self.test_microphone_recording():
            time.sleep(0.5) # A small pause to feel more natural
            self.speak("Hi, I'm Joey. I'm ready when you are.", 'en')
            while self.is_listening:
                try:
                    command = self.listen()
                    if command:
                        self.handle_command(command)
                except KeyboardInterrupt:
                    print("\n[JOEY] Shutting down on user request.")
                    self.is_listening = False
                except Exception as e:
                    print(f"[JOEY CRITICAL ERROR] An unexpected error occurred: {e}")
                    self.speak("Oops, something went wrong. I'm going to need a moment to reboot.", 'en')
                    self.is_listening = False

# --- Entry Point ---
if __name__ == "__main__":
    joey_assistant = Joey()
    joey_assistant.start()
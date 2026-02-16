# Joey: Your Smart In-Car AI Assistant (OkDriver) 

**Joey** is an offline-first, bilingual (English & Hindi) voice assistant designed specifically for drivers in India. Built as part of the **OkDriver** project, Joey aims to reduce driver distraction by providing hands-free control over vehicle status, navigation, and entertainment.



##  Key Features
* **Bilingual Support:** Seamlessly switch between English and Hindi.
* **Driver Utilities:** Voice commands for fuel levels, tire pressure, AC control, and headlights.
* **Smart Navigation:** Entity extraction to understand "Navigate to [Location]".
* **Entertainment:** In-built jokes, riddles, and facts to keep the driver engaged.
* **Offline Recognition:** Powered by Vosk for privacy and reliability in low-network areas.
* **TF-IDF Intent Recognition:** Uses Machine Learning to understand user intent even if the phrasing varies.

##  Tech Stack
- **Language:** Python 3.x
- **STT (Speech-to-Text):** Vosk API (Offline)
- **TTS (Text-to-Speech):** `pyttsx3` (English) & `gTTS` with `pygame` (Hindi)
- **NLU (Natural Language Understanding):** Scikit-learn (TF-IDF & Cosine Similarity)
- **Audio:** `sounddevice` and `numpy`

##  Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/OkDriver-Joey.git](https://github.com/YourUsername/OkDriver-Joey.git)
   cd OkDriver-Joey'''

##  Install Dependencies:
pip install -r requirements.txt

##  Download Vosk Models:

* **Download English model:** vosk-model-en-in-0.5
* **Download Hindi model:** vosk-model-hi-0.22
Extract them into your project folder and update the paths in joey_v4.py.

##  Run Joey:
python joey_v4.py

##  Sample Commands
* "Joey, how much fuel is left?"
* "हिंदी मोड चालू करो" (Switch to Hindi)
* "Navigate to Akshardham Temple"
* "Tell me a joke"

# Project Context
**Developed as a 3rd-year B.Tech CSE project at IILM University, focusing on Human-Computer Interaction (HCI) and Real-time AI applications.**

import Speech_recognition
import pyttsx3

def Speak(command):
    Voice = pyttsx3.init()
    Voice.say(command)
    Voice.runAndWait()

sr = Speech_recognition.Recognizer()
with speech_recognition.Microphone() as source2:
    print("Adjusting ambient noise...")
    sr.adjust_for_ambient_noise(source2, duration = 2)
    print("Start Speaking...")
    
    audio2 = sr.listen(source2)
    
    textT = sr.recognize_google(audio2)
    textT = textT.lower()

    print("Heard text: " + textT)

Speak(textT)
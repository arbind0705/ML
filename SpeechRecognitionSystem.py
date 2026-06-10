import speech_recognition as sr
import pyttsx3

Voice = pyttsx3.init()

def Speak(command):
    Voice = pyttsx3.init()
    Voice.say(command)
    Voice.runAndWait()

r = sr.Recognizer()
with sr.Microphone() as source:
    print("Adjusting ambient noise...")
    r.adjust_for_ambient_noise(source, duration = 2)
    print("Start Speaking...")
    
    audio2 = r.listen(source, timeout = 3)
    
    
    try:        
        textT = r.recognize_google(audio2)
        textT = textT.lower()
        print("Heard text: " + textT)
        Speak(textT)
    except sr.UnknownValueError:
        print("unable to recognize speech")
    except sr.RequestError:
        print("unable to connect xxxxx")

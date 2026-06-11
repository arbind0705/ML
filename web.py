import speech_recognition
import pyttsx3
import webbrowser

if __name__ == "__main__":

    sr = speech_recognition.Recognizer()

    with speech_recognition.Microphone() as source:
        sr.adjust_for_ambient_noise(source, duration=2)
        print("start speaking...")
        audio = sr.listen(source, timeout=3)
        print("listening...")

    try:
        destination = sr.recognize_google(audio).lower()
        print("Heard line...", destination)
        webbrowser.open(f"https://www.{destination}.com")

    except Exception as e:
        print("error : " + str(e))

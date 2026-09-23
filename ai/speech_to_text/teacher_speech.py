import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Use microphone
mic = sr.Microphone()

print("Speech-to-Text System Started")
print("Speak something... (Ctrl+C to stop)")

with mic as source:
    recognizer.adjust_for_ambient_noise(source)

while True:
    try:
        with mic as source:
            print("\nListening...")
            audio = recognizer.listen(source)

        print("Processing...")

        text = recognizer.recognize_google(audio)

        print("Teacher said:", text)

    except sr.UnknownValueError:
        print("Could not understand audio")

    except sr.RequestError as e:
        print("API error:", e)

    except KeyboardInterrupt:
        print("\nStopped.")
        break
import pyttsx3

confidence = 0.9
object = "roshy"

engine = pyttsx3.init()
if confidence > 0.8:
    text = f"I am {confidence*100:.0f} percent sure that this is a {object}."
else:
    text = f"I am somewhat confident this object is a {object}."

engine.say(text)
engine.runAndWait()

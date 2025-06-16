import random
import cv2
import numpy as np
import os
import pyttsx3
import serial
import time
import threading
from ultralytics import YOLO
import requests
import atexit
import face_recognition
# import json
from deepface import DeepFace  
import logging
import re
# import sqlite3

# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# from email.mime.base import MIMEBase
# from email import encoders
import pandas as pd
# from openpyxl import Workbook
# from openpyxl.drawing.image import Image as ExcelImage
# from PIL import Image
import datetime
import schedule
# from geopy.geocoders import Nominatim
import requests
# from bs4 import BeautifulSoup
from simple_pid import PID
from whisper_mic import WhisperMic


EXCEL_FILE = "detected_data.xlsx"


logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")





def save_to_excel(name, age, gender, emotion, health, image):
    """
    Save the detected data and image to an Excel file.
    """
    try:
        # Create a timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        
        name = re.sub(r'[\\/*?:"<>|]', "_", name) 
        timestamp = re.sub(r'[\\/*?:"<>|]', "_", timestamp)


        image_path = f"detected_images/{name}_{timestamp}.png"
        os.makedirs("detected_images", exist_ok=True)

  
        if not os.path.exists("detected_images"):
            logging.error("Failed to create 'detected_images' folder.")
            return


        if image is None:
            logging.error("Invalid image data. Cannot save image.")
            return

        try:
            cv2.imwrite(image_path, image)
        except Exception as e:
            logging.error(f"Failed to save image: {e}")
            return

        if not os.path.exists(EXCEL_FILE):
            df = pd.DataFrame(columns=["Timestamp", "Name", "Age", "Gender", "Emotion", "Health", "Image Path"])
        else:
            df = pd.read_excel(EXCEL_FILE)

        # Create a new DataFrame for the new data
        new_data = pd.DataFrame({
            "Timestamp": [timestamp],
            "Name": [name],
            "Age": [age],
            "Gender": [gender],
            "Emotion": [emotion],
            "Health": [health],
            "Image Path": [image_path]
        })

        df = pd.concat([df, new_data], ignore_index=True)


        df.to_excel(EXCEL_FILE, index=False)

        logging.info(f"Data saved to Excel: {new_data.iloc[0].to_dict()}")
    except Exception as e:
        logging.error(f"Error saving data to Excel: {e}")



logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")


try:
    arduino = serial.Serial(
        'COM3',          
        baudrate=115200,
        dsrdtr=None,     
        timeout=1
    )
    time.sleep(2)        
    # arduino.dtr = None             
    # arduino.reset_input_buffer()
except serial.SerialException as e:
    print(f"Failed to connect to COM3 (ESP32): {e}")
    arduino = None





try:
    arduino_temp_humidity = serial.Serial(
        'COM7', 
        2000000,
        timeout=1,
        write_timeout=None
    )
except serial.SerialException as e:
    print(f"Failed to connect to COM7 (temp/humidity): {e}")
    arduino_temp_humidity = None 







engine = pyttsx3.init()



model_path = "2.pt" 
model = YOLO(model_path).to('cuda') 

class_list = model.names


detection_colors = {i: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(len(class_list))}


frame_wid, frame_hyt = 290, 240  
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera. Check webcam connection.")


stop_event = threading.Event()

OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEEPSEEK_MODEL_NAME = "tinyllama"


known_face_encodings = []
known_face_names = []


def detect_health(frame):
    
    try:
        results = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        if results:
            emotion = results[0]["dominant_emotion"]
            if emotion in ["sad", "angry", "fear"]:
                return "Stress detected. Please take a break."
            elif emotion == "tired":
                return "Fatigue detected. Consider resting."
            else:
                return "Healthy"
    except Exception as e:
        logging.error(f"Error in health detection: {e}")
    return "Unknown"





def detect_health(frame):
    """
    Detect health conditions based on facial analysis.
    """
    try:
        results = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        if results:
            emotion = results[0]["dominant_emotion"]
            if emotion in ["sad", "angry", "fear"]:
                condition = "Stress detected. Please take a break."
                
                return condition
            elif emotion == "tired":
                condition = "Fatigue detected. Consider resting."
               
                return condition
            else:
                return "Healthy"
    except Exception as e:
        logging.error(f"Error in health detection: {e}")
    return "Unknown"



def get_medical_advice(symptoms):
   
    try:
        # Define the Ollama API endpoint
        ollama_url = "http://localhost:11434/api/generate"  

        
        payload = {
            "model": "tinyllama",  
            "prompt": f"Provide medical advice for the following symptoms: {symptoms}",
            "stream": False  
        }

        
        response = requests.post(ollama_url, json=payload)
        response.raise_for_status()  

       
        advice = response.json().get("response", "").strip()
        return advice

    except Exception as e:
        logging.error(f"Error fetching medical advice from Ollama: {e}")
        return "Sorry, I couldn't fetch medical advice."
    

    
def detect_age_gender_mood_health(frame, name=None):
    """
    Detect age, gender, mood, and health from a frame and store the data in the database and Excel sheet.
    """
    try:
        
        results = DeepFace.analyze(frame, actions=["age", "gender", "emotion"], enforce_detection=False)
        if results:
            result = results[0]  
            age = result.get("age", "unknown")
            gender = result.get("gender", "unknown")
            emotion = result.get("dominant_emotion", "unknown")
            health = "healthy" 

           
            save_to_excel(name, age, gender, emotion, health, frame)

            return age, gender, emotion, health
    except Exception as e:
        logging.error(f"Error in age/gender/mood detection: {e}")
    return None, None, None, None


BRAIN_TUMOR_MODEL_PATH = "brain_tumor_yolo.pt"  
brain_tumor_model = YOLO(BRAIN_TUMOR_MODEL_PATH).to('cuda') 


def analyze_brain_tumor(image_path):
  
    try:
    
        frame = cv2.imread(image_path)
        if frame is None:
            logging.error(f"Failed to load image: {image_path}")
            return None

        
        frame = cv2.resize(frame, (frame_wid, frame_hyt))

        
        results = brain_tumor_model.predict(frame, conf=0.45, save=False)

       
        if results:
            for box in results[0].boxes:
                conf = round(float(box.conf.cpu().numpy()[0]) * 100, 2)

                
                return {
                    "status": "Tumor Detected",
                    "confidence": conf
                }
        else:
            logging.warning("No tumor detected in the image.")
            return {
                "status": "No Tumor Detected",
                "confidence": 0.0
            }
    except Exception as e:
        logging.error(f"Error analyzing brain tumor: {e}")
        return None

def save_tumor_results_to_excel(image_path, tumor_results):
    """
    Save brain tumor detection results to a new sheet in the Excel file.
    Args:
        image_path (str): Path to the analyzed image.
        tumor_results (dict): Results from the tumor detection function.
    """
    try:
      
        if os.path.exists(EXCEL_FILE):
            workbook = pd.ExcelFile(EXCEL_FILE)
            sheets = workbook.sheet_names
            df = pd.read_excel(EXCEL_FILE, sheet_name=sheets[0])
        else:
            df = pd.DataFrame(columns=["Image Path", "Status", "Confidence"])

        
        new_data = pd.DataFrame({
            "Image Path": [image_path],
            "Status": [tumor_results["status"]],
            "Confidence": [tumor_results["confidence"]]
        })

       
        df = pd.concat([df, new_data], ignore_index=True)

        
        with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name="Brain Tumor Results", index=False)

        logging.info(f"Tumor detection results saved to Excel: {tumor_results}")
    except Exception as e:
        logging.error(f"Error saving tumor results to Excel: {e}")



def read_temperature_humidity():

    if arduino_temp_humidity is None:
        logging.error("Temperature/humidity Arduino is not connected.")
        return None, None

    try:
        if arduino_temp_humidity.in_waiting > 0:
            line = arduino_temp_humidity.readline().decode("utf-8").strip()
            if line.startswith("Temperature:") and ",Humidity:" in line:
                temp_str, humidity_str = line.split(",Humidity:")
                temperature = float(temp_str.replace("Temperature:", ""))
                humidity = float(humidity_str)
                return temperature, humidity
    except Exception as e:
        logging.error(f"Error reading from temperature/humidity Arduino: {e}")
    return None, None

def monitor_temperature_humidity():
   
    while not stop_event.is_set():
        temperature, humidity = read_temperature_humidity()
        if temperature is not None and humidity is not None:
            logging.info(f"Temperature: {temperature}°C, Humidity: {humidity}%")

            if temperature > 10:
                logging.warning("Temperature exceeded 50°C! Sending alert email.")
                # send_email(
                #     subject="High Temperature Alert",
                #     body=f"Temperature: {temperature}°C, Humidity: {humidity}%",
                #     to_email="ahmed2001365618200@gmail.com",  # Replace with recipient email
                #     attachment_path=None
                # )
        time.sleep(10)  # Check every 10 seconds
















def load_known_faces():
    known_faces_dir = "known_faces"
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0])





def generate_response(prompt, max_tokens=5, temperature=5):
    payload = {
        "model": DEEPSEEK_MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error communicating with Ollama API: {e}")
        return "Sorry, I couldn't generate a response."





pid_controllers = {
    # 0: PID(1, 0.1, 0.05, setpoint=90, output_limits=(60, 120)),  # Left shoulder
    # 1: PID(1, 0.1, 0.05, setpoint=130, output_limits=(100, 160)),  # Left elbow
    2: PID(1, 0.1, 0.05, setpoint=105, output_limits=(90, 140)), 
    3: PID(1, 0.1, 0.05, setpoint=150, output_limits=(60, 150)), 
   
    6: PID(1, 0.1, 0.05, setpoint=90, output_limits=(50, 90)),  
    7: PID(1, 0.1, 0.05, setpoint=150, output_limits=(60, 150))  
}


current_positions = {i: 90 for i in range(8)}  

def on_word(name, location, length):
    """Callback to sync mouth and gesture movement with speech."""
    try:
        # Open mouth (servo4) - no PID
        arduino_temp_humidity.write(b'servo4:130\n')
        
        
        target_positions = {
            # 0: random.randint(100, 160),
            # 1: random.randint(100, 160),
            2: random.randint(90, 140),
            3: random.randint(60, 150),
            # 5: random.randint(80, 130),
            6: random.randint(50, 90),
            7: random.randint(60, 150)
        }
        
     
        steps = 10
        for step in range(steps):
            commands = []
            for servo_id, pid in pid_controllers.items():
          
                pid.setpoint = target_positions[servo_id]
                
                
                new_pos = pid(current_positions[servo_id])
                current_positions[servo_id] = new_pos
                
                commands.append(f"servo{servo_id}:{int(new_pos)}")
            
            
            gesture_command = ";".join(commands) + "\n"
            arduino.write(gesture_command.encode())
            time.sleep(length/(100.0 * steps)) 
        
        # Close mouth
        arduino_temp_humidity.write(b'servo4:180\n')
        
    except Exception as e:
        print(f"Error in on_word: {e}")


def speak(text, emotion="neutral"):
   
    
    
    if emotion == "happy":
        rate = 160
        volume = 5
    elif emotion == "sad":
        rate = 160
        volume = 5
    else:
        rate = 160
        volume = 5

    engine.setProperty('rate', rate)
    engine.setProperty('volume', volume)

   
    engine.connect("started-word", on_word)

   
    engine.say(text)
    engine.runAndWait()









def listen_for_command():
    
    try:
        logging.info("Listening for voice commands via microphone...")
        
        mic = WhisperMic()
        
      
        result = mic.listen().strip()
        
        if result:
               green_color = '\033[92m'
               reset_color = '\033[0m'
               logging.info(f"Recognized command: {result}")
               print(f"{green_color}Recognized command: {result}{reset_color}")
               return result
               return ""
               
        
    except Exception as e:
        logging.error(f"Error in voice command processing: {e}")
        return ""




def count_objects():
    ret, frame = cap.read()
    if not ret:
        return 0, {}

    frame = cv2.resize(frame, (frame_wid, frame_hyt))
    results = model.predict(frame, conf=0.5, save=False) 

    object_counts = {}
    if results:
        for box in results[0].boxes:
            clsID = int(box.cls.cpu().numpy()[0])
            class_name = class_list[clsID].lower()
            if class_name in object_counts:
                object_counts[class_name] += 1
            else:
                object_counts[class_name] = 1

    return object_counts






def send_servo_command(servo_id, angle, speed):

    if servo_id < 0 or servo_id > 7:
        print(f"Invalid servo ID: {servo_id}. Must be between 0 and 7.")
        return
    if angle < 0 or angle > 180:
        print(f"Invalid angle: {angle}. Must be between 0 and 180.")
        return
    if speed < 0:
        print(f"Invalid speed: {speed}. Must be a positive value.")
        return

   
    command = f"servo{servo_id}:{angle}:{speed}\n"
    arduino.write(command.encode())  
    print(f"Sent to Arduino: {command.strip()}")

def both_hi():
    send_servo_command(0, 50, 15)
    send_servo_command(4, 50, 15)

    send_servo_command(1, 130, 15)
    send_servo_command(5, 130, 15)

    send_servo_command(3, 45, 15)
    send_servo_command(7, 45, 15)
    
    send_servo_command(2, 75, 15)
    send_servo_command(6, 75, 15)
     
    send_servo_command(2, 120, 15)
    send_servo_command(6, 120, 15)
    
    send_servo_command(2, 75, 15)  
    send_servo_command(6, 75, 15)  
    
    send_servo_command(2, 50, 15)  
    send_servo_command(6, 50, 15)  
    
    send_servo_command(2, 75, 15)
    send_servo_command(6, 75, 15)  
    
    
    send_servo_command(0, 100, 15)  
    send_servo_command(4, 100, 15)  
    
    send_servo_command(1, 130, 15)  
    send_servo_command(5, 130, 15) 
    
    send_servo_command(2, 75, 15)  
    send_servo_command(6, 75, 15)  
    
    send_servo_command(3, 155, 15)  
    send_servo_command(7, 155, 15)
     

def left_hi():
    send_servo_command(0, 50, 5)

    send_servo_command(1, 110, 5)

    send_servo_command(3, 45, 5)
    send_servo_command(2, 75, 5)
    send_servo_command(2, 120, 5)
    send_servo_command(2, 75, 5)  
    send_servo_command(2, 50, 5)  
    send_servo_command(2, 75, 5)
    
    
    send_servo_command(0, 100, 5)  
    send_servo_command(1, 130, 5)  
    send_servo_command(2, 75, 5)  
    send_servo_command(3, 155, 5)    

def right_hi():
    send_servo_command(4, 50, 15)

    send_servo_command(5, 130, 15)

    send_servo_command(7, 45, 15)
    send_servo_command(6, 75, 15)
    send_servo_command(6, 120, 15)
    send_servo_command(6, 75, 15)  
    send_servo_command(6, 50, 15)  
    send_servo_command(6, 75, 15)
    
    
    send_servo_command(4, 100, 15)  
    send_servo_command(5, 130, 15)  
    send_servo_command(6, 75, 15)  
    send_servo_command(7, 155, 15)    





def recognize_faces(frame):
   
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

            # Detect age, gender, emotion, and health
            age, gender, emotion, health = detect_age_gender_mood_health(frame, name)

        names.append(name)

    return names






def word_to_number(word):
   
    word_to_num = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10"
    }
    return word_to_num.get(word.lower(), word)  



external_process = None  




def perform_action(command):
    global external_process
    print(f"\nProcessing command: {command}")
   
    command_lower = command.lower()
    
    if "analyse the picture" in command_lower or "detect brain tumor picture" in command_lower:
        try:
          
            picture_number_word = re.search(r"(zero|one|two|three|four|five|six|seven|eight|nine|ten|\d+)", command_lower)
            if picture_number_word:
                picture_number_word = picture_number_word.group()
                picture_number = word_to_number(picture_number_word)
                image_path = f"brain_images/picture_{picture_number}.jpg"

                # Check if the image exists
                if not os.path.exists(image_path):
                    response = f"Picture {picture_number} not found."
                    logging.error(response)
                    speak(response, emotion="neutral")
                    return

              
                tumor_results = analyze_brain_tumor(image_path)
                if tumor_results:
                    response = (
                        f"Analysis results for picture {picture_number}: "
                        f"Status: {tumor_results['status']}, "
                        f"Confidence: {tumor_results['confidence']:.2f}%"
                    )
                    logging.info(response)
                    speak(response, emotion="neutral")

                   
                    save_tumor_results_to_excel(image_path, tumor_results)
                else:
                    response = "Sorry, I couldn't analyze the picture."
                    logging.error(response)
                    speak(response, emotion="neutral")
            else:
                response = "Please specify a picture number."
                logging.error(response)
                speak(response, emotion="neutral")
        except Exception as e:
            response = "Sorry, I couldn't process the picture."
            logging.error(f"Error analyzing picture: {e}")
            speak(response, emotion="neutral")
        return

   
    if "i feel" in command_lower:
        symptoms = command_lower.replace("i feel", "").strip()
        advice = get_medical_advice(symptoms)
        response = f"Based on your symptoms, here's my advice: {advice}"
        logging.info(response)
        speak(response, emotion="neutral")
        return

    elif "check my health" in command_lower:
        ret, frame = cap.read()
        if ret:
            age, gender, emotion, health = detect_age_gender_mood_health(frame)
            response = f"Your current health status is: {health}"
        else:
            response = "I couldn't capture the frame."
        logging.info(response)
        speak(response, emotion="neutral")
        return

    
    
    
    if "what is the temperature" in command_lower:
        temperature, humidity = read_temperature_humidity()
        if temperature is not None:
            response = f"The temperature is {temperature} degrees Celsius."
        else:
            response = "I couldn't read the temperature."
        logging.info(response)
        speak(response, emotion="neutral")
        return

    elif "what is the humidity" in command_lower:
        temperature, humidity = read_temperature_humidity()
        if humidity is not None:
            response = f"The humidity is {humidity} percent."
        else:
            response = "I couldn't read the humidity."
        logging.info(response)
        speak(response, emotion="neutral")
        return
    
    
    
    # if "start tracking drones" in command_lower:
    #     track_drone()
    #     response = "Starting drone tracking."
    #     logging.info(response)
    #     speak(response, emotion="happy")
    #     return  # Exit after handling the local command

    # elif "stop tracking drones" in command_lower:
    #     stop_drone_tracking()
    #     response = "Stopping drone tracking."
    #     logging.info(response)
    #     speak(response, emotion="neutral")
    #     return  # Exit after handling the local command

    # elif "give information about" in command_lower:
    #     name = command_lower.replace("give information about", "").strip()
    #     if name:
    #         person_info = get_person_info(name)
    #         if person_info:
    #             response = (
    #                 f"{name} was detected on {person_info['timestamp']}. "
    #                 f"They are approximately {person_info['age']} years old, "
    #                 f"{person_info['gender']}, and appear to be {person_info['emotion']}. "
    #                 f"Their health status is {person_info['health']}."
    #             )
    #         else:
    #             response = f"No information found for {name}."
    #     else:
    #         response = "Please specify a name."
    #     logging.info(response)
    #     speak(response, emotion="neutral")
    #     return  # Exit after handling the local command

    # elif "start tracking people" in command_lower:
    #     track_known_people()
    #     response = "Tracking dangerous people."
    #     logging.info(response)
    #     speak(response, emotion="neutral")
    #     return  # Exit after handling the local command

    elif "introduce yourself" in command_lower:
        left_hi()
        response = "Hi, I am Jarvis, a humanoid robot assistant developed by Ahmed Hassan under the supervision of Dr. Mohamed Khamry."
        logging.info(response)
        speak(response, emotion="sad")
        return  # Exit after handling the local command

    elif "how many people" in command_lower or "how many persons" in command_lower:
        object_counts = count_objects()
        count = object_counts.get("person", 0)
        response = f"I see {count} people."
        logging.info(response)
        speak(response, emotion="happy")
        return  

    elif "who are the persons" in command_lower or "who is here" in command_lower:
        ret, frame = cap.read()
        if ret:
            names = recognize_faces(frame)
            if names:
                response = "I see the following people: " + ", ".join(names)
            else:
                response = "I don't recognize anyone."
        else:
            response = "I couldn't capture the frame."
        logging.info(response)
        speak(response, emotion="neutral")
        return 

    elif "what objects you can see" in command_lower or "what do you see" in command_lower:
        object_counts = count_objects()
        if object_counts:
            response = "I see the following objects: "
            for obj, count in object_counts.items():
                response += f"{count} {obj}(s), "
            response = response.rstrip(", ") + "."
        else:
            response = "I don't see any objects."
        logging.info(response)
        speak(response, emotion="neutral")
        return 

    elif "guess the age" in command_lower or "guess the gender" in command_lower:
        ret, frame = cap.read()
        if ret:
            age, gender, emotion, health = detect_age_gender_mood_health(frame)
            if age and gender:
                response = f"I see a person who is approximately {age} years old. They appear to be {emotion}."
            else:
                response = "I couldn't detect the age or gender."
        else:
            response = "I couldn't capture the frame."
        logging.info(response)
        speak(response, emotion="neutral")
        return  # Exit after handling the local command

    elif "exit" in command_lower or "quit" in command_lower:
        speak("Goodbye!", emotion="sad")
        stop_event.set()
        cleanup()
        exit()

    
    response = generate_response(command)
    logging.info(f"Assistant: {response}")
    speak(response, emotion="sad")
        

def detect_emotion(frame):
    try:
        
        results = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        if results:
            emotion = results[0]["dominant_emotion"]
            return emotion
    except Exception as e:
        logging.error(f"Error in emotion detection: {e}")
    return None



def set_reminder(time, task):
    schedule.every().day.at(time).do(lambda: speak(f"Reminder: {task}", emotion="neutral"))
    return f"Reminder set for {time} to {task}."




def main():
    speak("i am ready sir ", emotion="happy")
    monitor_thread = threading.Thread(target=monitor_temperature_humidity)
    monitor_thread.daemon = True
    monitor_thread.start()
    
  

    

    while not stop_event.is_set():
       
       
        
        

        command= listen_for_command()
        if command:  
            if "martin" in command.lower():
                speak("Yes, I'm listening.", emotion="neutral")
                
                action_command = listen_for_command()
                if action_command:
                    perform_action(action_command)
                else:
                    speak("I didn't catch that. Can you repeat?", emotion="confused")
        else:
            speak("Listening...", emotion="neutral")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

   
    
    
    
    
def send_servo_angles(servoX, servoY):
    """Send servo angles to Arduino via serial communication, each on a separate line."""
    if arduino is not None:
        command_x = f"servo1:{servoX}\n"  
        command_y = f"servo2:{servoY-5}\n"
        command_z= f"servo2:{servoX-10}\n"  
        
        arduino_temp_humidity.write(command_x.encode())
        arduino_temp_humidity.write(command_z.encode())
        arduino_temp_humidity.write(command_y.encode()) 
        
        print(f"Sent to Arduino:\n{command_x.strip()}\n{command_y.strip()}")
    
    
    
    
    
    
    
    

target_class = "ahmed-hassan"  
tracking_active = False
proximity_threshold = 0.6  

def start_object_detection():
    global target_class, tracking_active
    
    try:
       
        servoPos = [90, 90]  

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue
            status_text = "waiting for target"
            status_color = (0, 255, 255) 

           

            frame = cv2.resize(frame, (frame_wid, frame_hyt))
            results = model.predict(frame, conf=0.5, save=False)

            target_detected = True
            closest_target_size = 0  
            
            if results and target_class: 
                for box in results[0].boxes:
                    clsID = int(box.cls.cpu().numpy()[0])
                    class_name = class_list[clsID].lower()
                    conf = round(float(box.conf.cpu().numpy()[0]) * 100, 2)
                    bb = box.xyxy.cpu().numpy()[0]

                    
                    color = detection_colors.get(clsID, (0, 255, 0))
                    if class_name == target_class:
                        color = (0, 0, 255) 
                        
                    cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), color, 3)
                    label = f"{class_list[clsID]}: {conf}%"
                    cv2.putText(frame, label, (int(bb[0]), int(bb[1]) - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                   
                    if class_name == target_class:
                        
                        target_width = bb[2] - bb[0]
                        target_height = bb[3] - bb[1]
                        target_size = target_width * target_height
                        
                        
                        if target_size > closest_target_size:
                            closest_target_size = target_size
                            target_detected = True
                            
                           
                            fx, fy = (int(bb[0]) + int(bb[2])) // 2, (int(bb[1]) + int(bb[3])) // 2
                            pos = [fx, fy]

                          
                            servoX = np.interp(fx, [0, frame_wid], [0, 180])
                            servoY = np.interp(fy, [0, frame_hyt], [180, 0])

                           
                            servoX = max(50, min(110, servoX))
                            servoY = max(50, min(100, servoY))

                            servoPos[0] = servoX
                            servoPos[1] = servoY

                            
                            cv2.circle(frame, (fx, fy), 5, (0, 0, 255), -1)
                            cv2.putText(frame, str(pos), (fx + 15, fy - 15), 
                                       cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                            cv2.line(frame, (0, fy), (frame_wid, fy), (0, 0, 0), 2)  
                            cv2.line(frame, (fx, frame_hyt), (fx, 0), (0, 0, 0), 2) 
                            status_text = f"TRACKING {target_class.upper()}"
                            cv2.putText(frame, status_text, (50, frame_hyt - 50), 
                                       cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                            
                            # Send servo commands
                            if arduino is not None:
                                command_all = f"servo1:{int(servoX)};servo2:{int(servoY) - 5};servo3:{int(servoX) - 10}\n"
                                arduino_temp_humidity.write(command_all.encode())
                                print(f"Sent to Arduino: {command_all.strip()}")

           
            if not target_detected:
                status_text = f"NO {target_class.upper()} DETECTED" if target_class else "NO TARGET SELECTED"
                status_color = (0, 0, 255) 
               
                cv2.circle(frame, (frame_wid // 2, frame_hyt // 2), 80, (0, 0, 255), 2)
                cv2.circle(frame, (frame_wid // 2, frame_hyt // 2), 15, (0, 0, 255), cv2.FILLED)
                cv2.line(frame, (0, frame_hyt // 2), (frame_wid, frame_hyt // 2), (0, 0, 0), 2)
                cv2.line(frame, (frame_wid // 2, frame_hyt), (frame_wid // 2, 0), (0, 0, 0), 2)
            else:
                status_color = (0, 255, 0) 

          
            cv2.putText(frame, status_text, (50, frame_hyt - 20), 
                       cv2.FONT_HERSHEY_PLAIN, 2, status_color, 2)
            cv2.putText(frame, f'Servo X: {int(servoPos[0])} deg', (50, 50), 
                       cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(frame, f'Servo Y: {int(servoPos[1])} deg', (50, 100), 
                       cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            cv2.imshow("Object Detection & Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

    finally:
        # arduino_temp_humidity.write(b'servo4:100\n')
        cv2.destroyAllWindows()





def cleanup():
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    if arduino and arduino.is_open:
        arduino.close()
    if external_process:
        external_process.terminate()
    stop_event.set()
    logging.info("Resources cleaned up.")

atexit.register(cleanup)


# a.daemon = False




if __name__ == "__main__":
    
    # arduino.write(b'0,0,180,40,100,180,0,40,0,0,0,0,0,0,0,0:25n')
    # time.sleep(8)
    # arduino.write(b'0,0,180,140,100,180,0,140,0,0,0,0,0,0,0,0:25n')
    # time.sleep(8)
    # arduino.write(b'100,130,105,150,100,100,90,150,0,0,0,0,0,0,0,0:25n')
    # time.sleep(8)
    # arduino.write(b'0,0,180,40,100,180,0,40,0,0,0,0,0,0,0,0:25n')
    # time.sleep(8)
    # arduino.write(b'0,0,180,140,100,180,0,140,0,0,0,0,0,0,0,0:25n')
    # time.sleep(8)
    # arduino.write(b'100,130,105,150,100,100,90,150,0,0,0,0,0,0,0,0:25n')
    # time.sleep(5)
    # make_decision()
    detection_thread = threading.Thread(target=start_object_detection)
    detection_thread.start() 
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\nExiting gracefully...")
    finally:
        cleanup()

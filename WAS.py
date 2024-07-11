import tkinter as tk
from tkinter import Frame, Label, Toplevel, Button
from PIL import Image, ImageTk
import cv2
import threading
import queue
import time
from ultralytics import YOLO
import mediapipe as mp
import pygame
import logging
import os  # Import für Verzeichnisauflistung
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import assemblyai as aai
import datetime
from datetime import datetime as dt
import os
from openai import OpenAI  # Importieren Sie die OpenAI-Klasse
import tkinter as tk

aai.settings.api_key = "XXXX"
is_recording = False
recording_thread = None
recording_start_time = None
recording_end_time = None
api_key = 'XXX'
client = OpenAI(api_key=api_key)

# Farben
RUB_BLAU = "#17365c"
RUB_GRUEN = "#8dae10"
RUB_GRAU = "#e7e7e7"

# Generieren des dynamischen Dateinamens für die Log-Datei
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"log_{timestamp}.txt"
log_directory = "C:/Users/aram/Desktop/Masterarbeit_Projekt/Werkerassistenzsystem"
log_file_path = os.path.join(log_directory, log_filename)

# Logging konfigurieren
logging.basicConfig(level=logging.DEBUG, filename=log_file_path, filemode='w',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Beispiel für allgemeines Logging
logging.debug("Dies ist eine Debug-Nachricht")

# Soundeffekte für beide Systeme
pygame.mixer.init()
sound_correct = pygame.mixer.Sound(r'C:\Users\aram\Desktop\Masterarbeit_Projekt\sounds\check_SoundFromZapsplat.mp3')
sound_complete = pygame.mixer.Sound(r'C:\Users\aram\Desktop\Masterarbeit_Projekt\sounds\complete_SoundFromZapsplat.mp3')
# Fehlerdefinitionen für falsch herum platzierte Teile
FEHLER_RING_FALSCH_HERUM = "Ring - falsch herum"
FEHLER_HALTETEIL_FALSCH_HERUM = "Halteteil - falsch herum"
# Mapping of detected names to expected names
name_mapping = {
    "O-Ring 20x1.5 -3-": "O-Ring 20x1.5 (3)"

}

# Soundeffekte für Fehler
sound_wrong = pygame.mixer.Sound(r'C:\Users\aram\Desktop\Masterarbeit_Projekt\sounds\wrong_SoundFromZapsplat.mp3')

# Fehlermeldungen
error_messages = {
    "wrong_order": "Fehler: Schritt übersprungen. Bitte erst den aktuellen Schritt abschließen.",
    FEHLER_RING_FALSCH_HERUM: "Fehler: Ring ist falsch herum platziert. Bitte umdrehen.",
    FEHLER_HALTETEIL_FALSCH_HERUM: "Fehler: Halteteil ist falsch herum platziert. Bitte umdrehen."
}


# Gemeinsame Konfigurationen
stop_threads = False  # Kontrolliert die Ausführung von Threads

# Teilsystem 1
model_path_1 = r'C:\Users\aram\Desktop\Masterarbeit_Projekt\ModelleV8\QualitätssicherungV8_2.pt'
model_1 = YOLO(model_path_1)
input_mode_1 = 'camera'  # Kann 'video' oder 'camera' sein
video_path_1 = r'C:\Users\aram\Desktop\Masterarbeit_Projekt\Videos\1.avi'  # Pfad zum Video für Teilsystem 1
cam_index_1 = 1  # Kamera-Index für Teilsystem 1
if input_mode_1 == 'camera':
    cap_1 = cv2.VideoCapture(cam_index_1, cv2.CAP_DSHOW)
    focus_value = 55
    cap_1.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap_1.set(cv2.CAP_PROP_FOCUS, focus_value)
else:
    cap_1 = cv2.VideoCapture(video_path_1)
    cap_1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_queue_1 = queue.Queue(maxsize=10)
box_coordinates_1 = {
    "Unterteil": {"x": 239, "y": 204, "width": 111, "height": 262},
    "1. O-Ring 12x2.5": {"x": 231, "y": 367, "width": 121, "height": 48},
    "Ring": {"x": 238, "y": 331, "width": 119, "height": 66},
    "2. O-Ring 12x2.5": {"x": 243, "y": 334, "width": 101, "height": 45},
    "Halteteil": {"x": 219, "y": 264, "width": 150, "height": 111},
    "O-Ring 20x1.5 (3)": {"x": 219, "y": 282, "width": 143, "height": 63},
    "Oberteil": {"x": 239, "y": 39, "width": 113, "height": 244},
    "Endmontage": {"x": 207, "y": 26, "width": 163, "height": 440}
}
# Feste Koordinaten für den Schraubendreher
box_coordinates_2 = {
    "Schraubendreherbox": {"x": 1180, "y": 350, "width": 100, "height": 100}  # Angepasste Koordinaten
}


# Teilsystem 2
model_path_2 = r'C:\Users\aram\Desktop\Masterarbeit_Projekt\ModelleV8\ObjekterkennungV8_3.pt'
model_2 = YOLO(model_path_2)
input_mode_2 = 'camera'  # 'video' oder 'camera'
video_path_2 = r'C:\Users\aram\Desktop\Masterarbeit_Projekt\Videos\test.mp4'
cam_index_2 = 0  # Ändern für Kamera!!
if input_mode_2 == 'camera':
    cap_2 = cv2.VideoCapture(cam_index_2, cv2.CAP_DSHOW)
    cap_2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Niedrigere Auflösung
    cap_2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
else:
    cap_2 = cv2.VideoCapture(video_path_2)
    cap_2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_queue_2 = queue.Queue(maxsize=10)
mp_hands_2 = mp.solutions.hands
hands_2 = mp_hands_2.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.2, min_tracking_confidence=0.1)

# Schrittfolge und Anleitungen
steps = [
    "Bitte ein (1) Unterteil aus der markierten Box entnehmen und wie auf dem Bild zu sehen in die Halterung legen",
    "Bitte einen (1) O-Ring (12x2.5) aus der markierten Box entnehmen und wie auf dem Bild zu sehen auf das Unterteil schieben",
    "Bitte einen (1) Ring aus der markierten Box entnehmen und wie auf dem Bild zu sehen auf das Unterteil schieben",
    "Bitte einen (1) O-Ring (12x2.5) aus der markierten Box entnehmen und wie auf dem Bild zu sehen auf das Unterteil schieben",
    "Bitte ein (1) Halteteil aus der markierten Box entnehmen und wie auf dem Bild zu sehen auf das Unterteil schrauben (Bitte links herum drehen)",
    "Bitte drei (3) O-Ringe  (20x1.5) aus der markierten Box entnehmen und wie auf dem Bild zu sehen auf das Halteteil spannen",
    "Bitte ein (1) Oberteil aus der markierten Box entnehmen und wie auf dem Bild zu sehen auf das Unterteil schieben",
    "Bitte eine (1) Schraube und den Schraubendreher aus den markierten Boxen nehmen und wie auf dem Bild zu sehen das Oberteil an das Unterteil schrauben. Legen Sie den Schraubendreher danach bitte wieder an seinen Platz. (Rechter Videstream kann hier ignoriert werden)",
    "Bauteile werden überprüft. Bitte warten...",
    "Montage fertig - Unilokk aus der Halterung nehmen und in die Transportbox vor Ihnen legen"
]

current_step_index = 0
montage_steps_2 = {
    "Unterteil": "Unterteile",
    "1. O-Ring 12x2.5": "O-Ringe 12x2.5",
    "Ring": "Ringe",
    "2. O-Ring 12x2.5": "O-Ringe 12x2.5",
    "Halteteil": "Halteteile",
    "O-Ring 20x1.5 (3)": "O-Ringe 20x1.5",
    "Oberteil": "Oberteile",
    "Oberteil": "Oberteil",
    "Schraube": "Schrauben",
    "Schraubendreher": "Schraubendreher",
    "Schraubendreherposition": "Schraubendreher",
    "Fertig": "Fertig"
}
current_step_name_2 = list(montage_steps_2.keys())[current_step_index]

#Flags:
ts1_detected = False
ts2_detected = False
ts2_detected_once = False
ts2_detected_schraube = False
ts2_detected_schraubendreher = False
second_screwdriver_detected = False
second_screwdriver_box_active = False

schraubendreher_timer_started = False
schraubendreher_start_time = 0

last_ts1_detect_time = time.time()

endmontage_timer_started = False
endmontage_start_time = 0

unilokk_counter = 0
start_time = time.time()

# Funktion für Farben
def get_color(class_name):
    colors = {
        "O-Ringe 12x2.5": (255, 0, 0),
        "1. O-Ring 12x2.5": (255, 0, 0),
        "O-Ringe 20x1.5": (0, 255, 0),
        "O-Ring 20x1.5 (3)": (0, 255, 0),
        "2. O-Ring 20x1.5": (0, 255, 0),
        "Schrauben": (0, 255, 255),
        "Unterteil": (255, 165, 0),
        "Unterteile": (255, 165, 0),
        "Halteteil": (160, 32, 240),
        "Halteteile": (160, 32, 240),
        "Ringe": (255, 105, 180),
        "Ring": (255, 105, 180),
        "Schraubendreher": (47, 79, 79),
        "Oberteil": (135, 206, 235),
        "Oberteile": (135, 206, 235),
        "Unilokk": (112, 128, 144)
    }
    return colors.get(class_name, (255, 255, 255))

# Funktion zum Erfassen und Anzeigen von Frames für Teilsystem 1
def capture_loop_1():
    global cap_1, frame_queue_1, ts1_detected, ts2_detected_once, ts2_detected_schraube, ts2_detected_schraubendreher
    global last_ts1_detect_time, endmontage_timer_started, endmontage_start_time

    while not stop_threads:
        ret, frame = cap_1.read()
        if not ret:
            logging.error("Teilsystem 1: Fehler beim Lesen des Frames")
            continue

        logging.debug("Teilsystem 1: Frame erfolgreich gelesen")

        if current_step_name_2 in ["Schraube", "Schraubendreherposition"]:
            logging.debug("Teilsystem 1: Erkennung gestoppt bei Schritt 'Schraube' oder 'Schraubendreherposition'")
        elif current_step_name_2 == "Endmontage":
            results = model_1(frame)
            if check_all_parts_present(results):
                if not endmontage_timer_started:
                    endmontage_timer_started = True
                    endmontage_start_time = time.time()
                elif time.time() - endmontage_start_time >= 2:
                    ts1_detected = True
                    advance_step()
            else:
                endmontage_timer_started = False
            frame = draw_boxes_1(frame, results)
        else:
            if current_step_index < len(box_coordinates_1):
                results = model_1(frame)
                frame = draw_boxes_1(frame, results)
                ts1_detected = check_detection_1(results)
                logging.debug(f"Teilsystem 1: ts1_detected={ts1_detected}, ts2_detected_once={ts2_detected_once}")

                if results[0].boxes:
                    last_ts1_detect_time = time.time()
                    if ts1_detected and ts2_detected_once and not check_for_errors(results):
                        advance_step()
                    else:
                        if check_for_errors(results):
                            ts1_detected = False  # Reset if an error is found
                else:
                    # Hier wird die Bedingung für den Schritt "Halteteil" hinzugefügt
                    if current_step_name_2 == "O-Ring 20x1.5 (3)":
                        reset_time = 30  # 30 Sekunden für den Schritt "Halteteil"
                    else:
                        reset_time = 5  # Standard 5 Sekunden für andere Schritte
                    if time.time() - last_ts1_detect_time > reset_time:
                        reset_to_first_step()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not frame_queue_1.full():
            frame_queue_1.put(frame)
        else:
            logging.warning("Teilsystem 1: Frame-Queue ist voll, aktueller Frame wird verworfen")

        time.sleep(0.03)


def check_for_errors(results):
    global current_step_index, steps, ts1_detected
    if results is None:
        logging.error("Teilsystem 1: Keine Ergebnisse erhalten")
        return False

    detected_parts = [results[0].names[int(box.cls[0].item())] for box in results[0].boxes]
    logging.debug(f"Erkannte Teile: {detected_parts}")

    if ts1_detected and current_step_index < len(steps) - 1:
        next_part_index = current_step_index + 1
        if next_part_index < len(box_coordinates_1):
            next_part = list(box_coordinates_1.keys())[next_part_index]
            if next_part in detected_parts:
                error_message = error_messages["wrong_order"]
                update_feedback(error_message, "red")
                sound_wrong.play()
                logging.debug(error_message)
                return True
        else:
            logging.error(f"Index {next_part_index} ist außerhalb des gültigen Bereichs für box_coordinates_1")

    for part in detected_parts:
        if part in error_messages:
            if (part == FEHLER_RING_FALSCH_HERUM and "Ring" in detected_parts) or \
               (part == FEHLER_HALTETEIL_FALSCH_HERUM and "Halteteil" in detected_parts):
                logging.debug(f"{part} erkannt, aber ignoriert, da das korrekte Teil ebenfalls erkannt wurde.")
                continue
            error_message = error_messages[part]
            update_feedback(error_message, "red")
            sound_wrong.play()
            logging.debug(error_message)
            return True

    return False



# Funktion zum Erfassen und Anzeigen von Frames für Teilsystem 2
def capture_loop_2():
    global cap_2, frame_queue_2, ts1_detected, ts2_detected_once, ts2_detected_schraube, ts2_detected_schraubendreher, schraubendreher_timer_started, schraubendreher_start_time, second_screwdriver_box_active

    while not stop_threads:
        ret, frame = cap_2.read()

        if ret:
            logging.debug("Teilsystem 2: Frame erfolgreich gelesen")
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            results = model_2(frame)
            detected_parts = [results[0].names[int(box.cls.item())] for box in results[0].boxes]

            logging.debug(f"Teilsystem 2: Erkannte Teile: {detected_parts}")

            if current_step_name_2 != "Endmontage":
                frame = draw_boxes_2(frame, results)

            if current_step_name_2 == "Schraube":
                frame = draw_fixed_box_2(frame)
                process_mediapipe_results_2(frame)

                if ts2_detected_schraube and ts2_detected_schraubendreher:
                    advance_step()

            elif current_step_name_2 == "Schraubendreherposition":
                if (time.time() - schraubendreher_start_time) >= 3:
                    second_screwdriver_box_active = True
                if second_screwdriver_box_active:
                    frame = show_second_screwdriver_box(frame)
                    process_mediapipe_results_2(frame)
                    if second_screwdriver_detected:
                        advance_step()

            elif current_step_name_2 == "Endmontage":
                # Teilsystem 2 bei Endmontage nicht aktiv
                pass

            else:
                if process_mediapipe_results_2(frame):
                    ts2_detected_once = True

                logging.debug(
                    f"Teilsystem 2: ts1_detected={ts1_detected}, ts2_detected={ts2_detected}, ts2_detected_once={ts2_detected_once}")

                if ts1_detected and ts2_detected_once:
                    advance_step()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not frame_queue_2.full():
            frame_queue_2.put(frame)

        time.sleep(0.03)


# Funktion zum Zeichnen von Boxen für Teilsystem 1
def draw_boxes_1(frame, results):
    global current_step_index, box_coordinates_1, current_step_name_2
    if current_step_name_2 != "Fertig" and current_step_index < len(box_coordinates_1):
        part = list(box_coordinates_1.keys())[current_step_index]
        box = box_coordinates_1[part]
        color = get_color_based_on_detection_1(ts1_detected)
        cv2.rectangle(frame, (box["x"], box["y"]), (box["x"] + box["width"], box["y"] + box["height"]), color, 2)
        cv2.putText(frame, part, (box["x"], box["y"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        logging.debug(f"Teilsystem 1: Box für {part} gezeichnet mit Farbe {color}")
    return frame


# Funktion zum Zeichnen von Boxen für Teilsystem 2
def draw_boxes_2(frame, results):
    global results_2, ts2_detected_schraube, montage_steps_2, current_step_name_2
    results_2 = results

    if current_step_name_2 not in montage_steps_2 or current_step_name_2 == "Fertig":
        return frame

    for box in results[0].boxes:
        class_name = results[0].names[int(box.cls.item())]
        xyxy = box.xyxy.detach().cpu().numpy().squeeze()
        if class_name == montage_steps_2.get(current_step_name_2):
            if current_step_name_2 == "Schraube":
                color = (0, 255, 0) if ts2_detected_schraube else (255, 0, 0)
            else:
                color = (0, 255, 0) if ts2_detected else (255, 0, 0)
            label = f'{class_name} {box.conf.item():.2f}'
            frame = plot_one_box_2(xyxy, frame, label, color)
            logging.debug(f"Teilsystem 2: Box für {class_name} gezeichnet mit Farbe {color}")
    return frame


# Funktion zum Ändern der Farbe basierend auf dem Status der Erkennung
def get_color_based_on_detection_1(detected):
    return (0, 255, 0) if detected else (0, 0, 255)

# Funktion zum Ändern der Farbe basierend auf dem Status der Erkennung
def get_color_based_on_detection_2(detected):
    return (0, 255, 0) if detected else (255, 0, 0)


# Funktion zum Zeichnen einer Box

def plot_one_box_2(xyxy, frame, label=None, color=(128, 128, 128), line_thickness=3):
    tl = line_thickness or round(0.002 * (frame.shape[0] + frame.shape[1]) / 2) + 1
    color = color[::-1]
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(frame, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(frame, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(frame, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return frame

# Globale Definition von results_2
results_2 = None


# Funktion zur Verarbeitung der MediaPipe-Ergebnisse für Teilsystem 2
def process_mediapipe_results_2(frame):
    global hands_2, ts2_detected, ts2_detected_schraube, ts2_detected_schraubendreher, current_step_name_2, second_screwdriver_detected, second_screwdriver_box_active
    frame_height, frame_width = frame.shape[:2]
    results = hands_2.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands_2.HAND_CONNECTIONS)
            logging.debug("Teilsystem 2: Handlandmark gezeichnet")
            if current_step_name_2 == "Schraube":
                if is_hand_in_yolo_box(hand_landmarks, frame_width, frame_height, frame):
                    logging.debug("Teilsystem 2: Hand in der YOLO Box erkannt")
                    ts2_detected_schraube = True
                if is_hand_in_fixed_box(hand_landmarks, frame_width, frame_height):
                    logging.debug("Teilsystem 2: Hand in der festen Box erkannt")
                    ts2_detected_schraubendreher = True
            elif second_screwdriver_box_active:
                if is_hand_in_fixed_box(hand_landmarks, frame_width, frame_height):
                    logging.debug("Teilsystem 2: Hand in der zweiten festen Box erkannt")
                    second_screwdriver_detected = True
            else:
                if is_hand_in_yolo_box(hand_landmarks, frame_width, frame_height, frame):
                    ts2_detected = True
    return ts2_detected



# Funktion zum Überprüfen, ob die Hand in einer YOLO-Box ist
def is_hand_in_yolo_box(hand_landmarks, frame_width, frame_height, frame):
    global results_2, current_step_name_2
    if results_2 is None:
        logging.debug("results_2 ist None, keine Box zum Überprüfen vorhanden.")
        return False

    hand_in_box = False
    for box in results_2[0].boxes:
        class_name = results_2[0].names[int(box.cls.item())]
        if class_name != montage_steps_2[current_step_name_2]:
            continue

        xyxy = box.xyxy.detach().cpu().numpy().squeeze()
        logging.debug(f"YOLO Box Koordinaten: {xyxy}")

        for idx, landmark in enumerate(hand_landmarks.landmark):
            lx, ly = int(landmark.x * frame_width), int(landmark.y * frame_height)
            logging.debug(f"Hand Landmark {idx}: (x={lx}, y={ly})")
            if xyxy[0] <= lx <= xyxy[2] and xyxy[1] <= ly <= xyxy[3]:
                logging.debug(
                    f"Handteil erkannt bei (x={lx}, y={ly}) in YOLO Box (x1={xyxy[0]}, y1={xyxy[1]}, x2={xyxy[2]}, y2={xyxy[3]})")
                hand_in_box = True
                break

    if not hand_in_box:
        logging.debug("Kein Teil der Hand in der Box erkannt")

    return hand_in_box

def is_hand_in_fixed_box(hand_landmarks, frame_width, frame_height):
    box = box_coordinates_2["Schraubendreherbox"]
    for landmark in hand_landmarks.landmark:
        lx, ly = int(landmark.x * frame_width), int(landmark.y * frame_height)
        if box["x"] <= lx <= box["x"] + box["width"] and box["y"] <= ly <= box["y"] + box["height"]:
            return True
    return False



# Funktion zum Zeichnen von Boxen für Teilsystem 1
def draw_boxes_1(frame, results):
    global current_step_index, box_coordinates_1, current_step_name_2
    if current_step_name_2 != "Fertig" and current_step_name_2 != "Endmontage" and current_step_index < len(box_coordinates_1):
        part = list(box_coordinates_1.keys())[current_step_index]
        box = box_coordinates_1[part]
        color = get_color_based_on_detection_1(ts1_detected)
        cv2.rectangle(frame, (box["x"], box["y"]), (box["x"] + box["width"], box["y"] + box["height"]), color, 2)
        cv2.putText(frame, part, (box["x"], box["y"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        logging.debug(f"Teilsystem 1: Box für {part} gezeichnet mit Farbe {color}")
    return frame


# Funktion zum Zeichnen von Boxen für Teilsystem 2
def draw_boxes_2(frame, results):
    global results_2, ts2_detected_schraube, montage_steps_2, current_step_name_2
    results_2 = results

    # Füge eine Überprüfung hinzu, ob current_step_name_2 "Endmontage" ist
    if current_step_name_2 == "Endmontage":
        return frame

    for box in results[0].boxes:
        class_name = results[0].names[int(box.cls.item())]
        xyxy = box.xyxy.detach().cpu().numpy().squeeze()
        if class_name == montage_steps_2[current_step_name_2]:
            if current_step_name_2 == "Schraube":
                color = (0, 255, 0) if ts2_detected_schraube else (255, 0, 0)
            else:
                color = (0, 255, 0) if ts2_detected else (255, 0, 0)
            label = f'{class_name} {box.conf.item():.2f}'
            frame = plot_one_box_2(xyxy, frame, label, color)
            logging.debug(f"Teilsystem 2: Box für {class_name} gezeichnet mit Farbe {color}")
    return frame


def draw_fixed_box_2(frame):
    global schraubendreher_timer_started, schraubendreher_start_time, ts2_detected_schraubendreher
    box = box_coordinates_2["Schraubendreherbox"]
    color = (0, 0, 255)  # Rot, bis die Hand erkannt wird
    if ts2_detected_schraubendreher:
        color = (0, 255, 0)  # Grün, wenn die Hand erkannt wird
    cv2.rectangle(frame, (box["x"], box["y"]), (box["x"] + box["width"], box["y"] + box["height"]), color, 2)
    cv2.putText(frame, "Schraubendreher", (box["x"] - 130, box["y"] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    logging.debug(f"Teilsystem 2: Schraubendreherbox gezeichnet mit Farbe {color}")
    return frame

def show_second_screwdriver_box(frame):
    global second_screwdriver_detected, second_screwdriver_box_active
    if second_screwdriver_box_active:
        box = box_coordinates_2["Schraubendreherbox"]
        color = (0, 0, 255)  # Rot, bis die Hand erkannt wird
        if second_screwdriver_detected:
            second_screwdriver_box_active = False  # Box deaktivieren, nachdem sie grün geworden ist
            return frame  # Box wird nicht mehr gezeichnet
        cv2.rectangle(frame, (box["x"], box["y"]), (box["x"] + box["width"], box["y"] + box["height"]), color, 2)
        cv2.putText(frame, "Schraubendreher", (box["x"] - 130, box["y"] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        logging.debug(f"Teilsystem 2: Zweite Schraubendreherbox gezeichnet mit Farbe {color}")
    return frame


# Funktion zum Überprüfen der Erkennung in Teilsystem 1
def check_detection_1(results):
    global box_coordinates_1, current_step_index, ts1_detected
    part = list(box_coordinates_1.keys())[current_step_index]
    box = box_coordinates_1[part]

    for detected_box in results[0].boxes:
        class_name = results[0].names[int(detected_box.cls.item())]
        if class_name in name_mapping:
            class_name = name_mapping[class_name]
        logging.debug(f"Erkanntes Teil: {class_name}, Erwartetes Teil: {part}")

        if class_name == part:
            xyxy = detected_box.xyxy.detach().cpu().numpy().squeeze()
            if is_in_box(xyxy, box):
                logging.debug(f"Teilsystem 1: {part} erkannt und in der richtigen Box positioniert.")
                ts1_detected = True
                return True

    ts1_detected = False
    return False

# Funktion zum Überprüfen, ob die erkannte Box innerhalb der definierten Box ist
def is_in_box(detected_box, reference_box):
    x1, y1, x2, y2 = detected_box
    box_x, box_y, box_w, box_h = reference_box["x"], reference_box["y"], reference_box["width"], reference_box["height"]

    # Überprüfen, ob die erkannte Box innerhalb der Referenzbox liegt
    if (box_x <= x1 <= box_x + box_w and box_y <= y1 <= box_y + box_h) and \
       (box_x <= x2 <= box_x + box_w and box_y <= y2 <= box_y + box_h):
        return True
    return False

# Funktion zum Überprüfen der Erkennung in Teilsystem 2
def check_detection_2():
    global box_coordinates_1, current_step_index
    # Überprüfen, ob die Hand in der richtigen Box ist
    return ts2_detected

# Funktion zum Voranschreiten des Schritts
def advance_step():
    global current_step_index, current_step_name_2, ts1_detected, ts2_detected, ts2_detected_once
    global ts2_detected_schraube, ts2_detected_schraubendreher, schraubendreher_timer_started
    global schraubendreher_start_time, second_screwdriver_detected, second_screwdriver_box_active
    global endmontage_timer_started, endmontage_start_time, sound_complete, unilokk_counter  # Ensure these are global

    logging.debug(
        f"advance_step aufgerufen. Aktueller Schritt: {current_step_index}, ts1_detected: {ts1_detected}, ts2_detected_once: {ts2_detected_once}")

    if current_step_name_2 == "Schraube":
        if ts2_detected_schraube and ts2_detected_schraubendreher:
            update_feedback(steps[current_step_index], RUB_GRUEN)
            sound_correct.play()
            ts1_detected = False
            ts2_detected = False
            ts2_detected_once = False
            ts2_detected_schraube = False
            ts2_detected_schraubendreher = False
            schraubendreher_timer_started = True
            schraubendreher_start_time = time.time()
            current_step_name_2 = "Schraubendreherposition"
            second_screwdriver_box_active = False

    elif current_step_name_2 == "Schraubendreherposition":
        if second_screwdriver_detected:
            second_screwdriver_box_active = False  # Deactivate the box after detection
            current_step_name_2 = "Endmontage"  # Move directly to Endmontage
            update_feedback("Bauteile werden überprüft. Bitte warten...", RUB_GRUEN)
            show_instruction_image(9)  # Zeige das Bild für Schritt 9
            sound_correct.play()
            reset_flags()

    elif current_step_name_2 == "Endmontage":
        if ts1_detected:
            update_feedback("Montage fertig - Unilokk aus der Halterung nehmen und in die Transportbox vor Ihnen legen",
                            RUB_GRUEN)
            show_instruction_image(10)  # Zeige das Bild für Schritt 10
            sound_complete.play()
            unilokk_counter += 1
            logging.debug(f"Unilokk Counter erhöht: {unilokk_counter}")
            unilokk_counter_label.config(text=f"Fertige Unilokks: {unilokk_counter}")
            reset_flags()
            current_step_name_2 = "Fertig"

    elif current_step_name_2 == "Fertig":
        update_feedback("Montage abgeschlossen", RUB_GRUEN)
        instruction_label.config(image='')  # Keine Box im letzten Schritt
        logging.debug("Montage abgeschlossen")

    else:
        current_step_index += 1
        if current_step_index < len(steps):
            current_step_name_2 = list(montage_steps_2.keys())[current_step_index]
            logging.debug(f"Neuer Schritt: {current_step_name_2}, Erwarteter Schritt: {steps[current_step_index]}")
            update_feedback(steps[current_step_index], RUB_GRUEN)
            show_instruction_image(current_step_index + 1)
            sound_correct.play()
            reset_flags()
        else:
            update_feedback("Montage abgeschlossen", RUB_GRUEN)
            instruction_label.config(image='')
            logging.debug("Montage abgeschlossen")

def reset_to_first_step():
    global current_step_index, current_step_name_2, ts1_detected, ts2_detected, ts2_detected_once, ts2_detected_schraube, ts2_detected_schraubendreher, schraubendreher_timer_started, second_screwdriver_detected, second_screwdriver_box_active, last_ts1_detect_time
    current_step_index = 0
    current_step_name_2 = list(montage_steps_2.keys())[current_step_index]
    logging.debug(f"Reset auf ersten Schritt: {current_step_name_2}, Erwarteter Schritt: {steps[current_step_index]}")
    update_feedback(steps[current_step_index], RUB_GRUEN)
    show_instruction_image(current_step_index + 1)
    reset_flags()
    last_ts1_detect_time = time.time()

def reset_flags():
    global ts1_detected, ts2_detected, ts2_detected_once, ts2_detected_schraube, ts2_detected_schraubendreher, schraubendreher_timer_started, second_screwdriver_detected, second_screwdriver_box_active
    ts1_detected = False
    ts2_detected = False
    ts2_detected_once = False
    ts2_detected_schraube = False
    ts2_detected_schraubendreher = False
    schraubendreher_timer_started = False
    second_screwdriver_detected = False
    second_screwdriver_box_active = False


#MUSS GEÄNDERT WERDEN FÜR ALLE TEILE
def check_all_parts_present(results):
    """Prüft, ob das Unterteil und das Oberteil vorhanden sind."""
    required_parts = ["Unterteil", "Oberteil"]
    detected_parts = [results[0].names[int(box.cls[0].item())] for box in results[0].boxes]
    for part in required_parts:
        if part not in detected_parts:
            return False
    return True



# Funktion zum Anzeigen des Anweisungsbildes
def show_instruction_image(step_number):
    try:
        image_path = f'C:/Users/aram/Desktop/Masterarbeit_Projekt/Bild/{step_number}.jpg'
        logging.debug(f"Versuche, Bild zu laden: {image_path}")
        image = Image.open(image_path)
        image = image.resize((300, 200), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        instruction_label.config(image=photo)
        instruction_label.image = photo
        logging.debug(f"Bild für Schritt {step_number} erfolgreich geladen")
    except Exception as e:
        logging.error(f"Fehler beim Laden des Bildes für Schritt {step_number}: {e}")

# Funktion zur Verzeichnisauflistung für Debugging
def list_image_directory():
    directory = 'C:/Users/aram/Desktop/Masterarbeit_Projekt/Bild/'
    try:
        files = os.listdir(directory)
        logging.debug(f"Dateien im Verzeichnis {directory}: {files}")
    except Exception as e:
        logging.error(f"Fehler beim Auflisten des Verzeichnisses {directory}: {e}")

# Verzeichnisinhalt beim Start auflisten
list_image_directory()

# Funktion zum Aktualisieren des Kamerabildes
def update_camera(frame, camera_label):
    if frame is not None:
        frame = cv2.resize(frame, (800, 600))
        img = ImageTk.PhotoImage(image=Image.fromarray(frame))
        camera_label.config(image=img)
        camera_label.image = img

# Funktion zum Aktualisieren des Kamerabildes aus der Warteschlange
def update_camera_from_queue(frame_queue, camera_label):
    if not frame_queue.empty():
        frame = frame_queue.get()
        update_camera(frame, camera_label)
    root.after(30, update_camera_from_queue, frame_queue, camera_label)

# Funktion zum Schließen der Anwendung
def on_closing():
    global stop_threads
    stop_threads = True
    if cap_1.isOpened():
        cap_1.release()
    if cap_2.isOpened():
        cap_2.release()
    cv2.destroyAllWindows()
    root.destroy()

# Start-Funktion, um das Hauptfenster anzuzeigen und das System zu initialisieren
def start_system():
    welcome_window.destroy()
    root.deiconify()
    show_instruction_image(1)  # Zeige das Bild für den ersten Schritt

def update_feedback(message, color):
    """Aktualisiert das Feedback basierend auf der übergebenen Nachricht und Farbe."""
    logging.debug(f"Feedback: {message} - Farbe: {color}")
    feedback_label.config(text=message, fg=color)

def update_timer():
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    work_time_label.config(text=f"Arbeitszeit: {hours:02}:{minutes:02}:{seconds:02}")
    logging.debug(f"Timer aktualisiert: {hours:02}:{minutes:02}:{seconds:02}")
    root.after(1000, update_timer)  # Timer jede Sekunde aktualisieren

def start_system():
    global start_time
    welcome_window.destroy()
    root.deiconify()
    start_time = time.time()
    update_timer()
    logging.debug("System gestartet. Timer läuft.")
    show_instruction_image(1)  # Zeige das Bild für den ersten Schritt

def start_recording():
    global is_recording, recording_thread, recording_start_time
    is_recording = True
    recording_start_time = datetime.datetime.now()
    recording_thread = threading.Thread(target=record_audio)
    recording_thread.start()

def stop_recording():
    global is_recording, recording_thread, recording_end_time
    is_recording = False
    recording_end_time = datetime.datetime.now()
    if recording_thread is not None:
        recording_thread.join()
        recording_thread = None
def record_audio():
    global recording_start_time, recording_end_time
    samplerate = 44100
    duration = 10  # seconds
    start_timestamp = recording_start_time.strftime("%Y%m%d_%H%M%S")
    filename = f"output_{start_timestamp}.wav"
    print("Recording...")
    myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=2, dtype='int16')
    sd.wait()  # Wait until recording is finished
    wav.write(filename, samplerate, myrecording)
    recording_end_time = datetime.datetime.now()  # Update the end time after recording is finished
    print("Recording complete. File saved as", filename)
    transcribe_audio(filename)


import assemblyai as aai

def transcribe_audio(file_path):
    global recording_start_time, recording_end_time
    transcriber = aai.Transcriber()
    # Set the language code to German
    config = aai.TranscriptionConfig(language_code="de")
    transcript = transcriber.transcribe(file_path, config=config)
    print(transcript.text)
    start_timestamp = recording_start_time.strftime("%Y%m%d_%H%M%S")
    end_timestamp = recording_end_time.strftime("%Y%m%d_%H%M%S")
    transcript_filename = f"transcription_{start_timestamp}.txt"
    with open(transcript_filename, "w") as f:
        f.write(f"Start time: {recording_start_time}\n")
        f.write(f"End time: {recording_end_time}\n")
        f.write(transcript.text)
    print("Transcription complete. File saved as", transcript_filename)

def toggle_recording():
    global is_recording
    if is_recording:
        stop_recording()
        record_button.config(text="Starte Sprachaufnahme")
        recording_status_label.config(text="")
    else:
        start_recording()
        record_button.config(text="Beende Sprachaufnahme")
        recording_status_label.config(text="Sprache wird aufgenommen...")


# Funktion zum Öffnen des Bildes in einem neuen Fenster
def show_help_image():
    help_window = Toplevel(root)
    help_window.title("Hilfe")


    image_path = r'C:\Users\aram\Desktop\Masterarbeit_Projekt\Bild\Unbenannt.JPG'
    image = Image.open(image_path)
    photo = ImageTk.PhotoImage(image)

    label = Label(help_window, image=photo)
    label.image = photo  # Referenz speichern, um das Bild anzuzeigen
    label.pack()

def analyze_content_with_chatgpt(content):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Du bist ein professioneller Assistent, der sich auf die Analyse von Log- und Transkriptdateien spezialisiert hat."
            },
            {
                "role": "user",
                "content": f"Bitte analysiere den folgenden Inhalt und fasse die wichtigsten Informationen zusammen:\n{content}\nFragen zur Analyse:\n- Wann wurde was gesagt?\n- Wann wurde welcher Schritt erfüllt?\n- Wie lange hat der Werker gebraucht?\n- Welcher Schritt dauerte am längsten?\n- Gab es Probleme?\n- Wurden Fehler ausgespuckt?\n- Wie oft gab es Fehler?"
            }
        ]
    )
    return response.choices[0].message.content

def analyze_files():
    folder_path = "C:/Users/aram/Desktop/Masterarbeit_Projekt/Werkerassistenzsystem"
    today_str = dt.now().strftime("%Y%m%d")

    log_files = [f for f in os.listdir(folder_path) if f.startswith(f'log_{today_str}') and f.endswith('.txt')]
    transcript_files = [f for f in os.listdir(folder_path) if f.startswith(f'transcription_{today_str}') and f.endswith('.txt')]

    analysis_results = []

    print(f"Found log files: {log_files}")  # Debugging-Information
    print(f"Found transcription files: {transcript_files}")  # Debugging-Information

    for file_name in log_files + transcript_files:
        file_path = os.path.join(folder_path, file_name)
        print(f"Analyzing file: {file_path}")  # Debugging-Information
        with open(file_path, 'r') as file:
            content = file.read()
            print(f"Content of {file_path}: {content[:100]}...")  # Print the first 100 characters for debugging
            response = analyze_content_with_chatgpt(content)
            analysis_results.append(f"Analyse für {file_name}:\n{response}\n")

    # Erstellen des Dateinamens mit Datum und Zeitstempel
    timestamp_str = dt.now().strftime("%Y-%m-%d_%H%M%S")
    result_file_name = f"ANALYSE_{timestamp_str}.txt"

    # Ergebnisse in einer Textdatei speichern
    with open(os.path.join(folder_path, result_file_name), 'w') as result_file:
        result_file.write("\n".join(analysis_results))
    print(f"Analyseergebnisse gespeichert in {result_file_name}")



# Funktion zum Schließen der Anwendung bei Escape-Taste
def close_on_escape(event=None):
    on_closing()

# GUI-Setup
root = tk.Tk()
root.title("WERKERASSISTENZSYSTEM")

# Setze die Fenstergröße relativ zur Bildschirmgröße
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = int(screen_width * 0.8)
window_height = int(screen_height * 0.8)
root.geometry(f"{window_width}x{window_height}")
root.configure(bg="#17365c")
root.withdraw()

# Plattform spezifische Maximierung
root.update_idletasks()
root.attributes('-fullscreen', True)
root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))

# Willkommen-Fenster
welcome_window = Toplevel(root)
welcome_window.title("Willkommen")
welcome_window.attributes('-fullscreen', True)
welcome_window.bind("<Escape>", lambda e: welcome_window.attributes("-fullscreen", False))

# Setze den Hintergrund direkt beim Erstellen der Widgets
welcome_window.configure(bg="#17365c")  # Hintergrundfarbe setzen
welcome_window.configure(highlightbackground="#17365c", highlightcolor="#17365c")  # Randfarben setzen

welcome_label = Label(welcome_window, text="WILLKOMMEN", font=('Helvetica', 18, 'bold'), fg="#e7e7e7", bg="#17365c")
welcome_label.pack(pady=20)
info_frame = tk.Frame(welcome_window, bg="#17365c")
info_frame.pack(pady=20, padx=20, fill='both', expand=True)

info_texts = [
    "Willkommen beim Werkerassistenzsystem zum Bau eines Unilokks.",
    "",
    "Dieses System führt Sie Schritt für Schritt durch den Aufbau eines Unilokks.",
    "",
    "Anweisungen:",
    "- Oben im Infofeld wird genau beschrieben, was zu tun ist, zusammen mit einem Beispielbild, an dem Sie sich orientieren können.",
    "- Die Benutzeroberfläche zeigt beide Videostreams an.",
    "",
    "Kameras:",
    "1. Die erste Kamera ist von oben auf die Montagestation gerichtet. Sie zeigt, wo Sie die Bauteile entnehmen müssen. Führen Sie die Hand langsam in das markierte Feld, damit das System Ihre Bewegung erkennen kann. Sobald Ihre Hand erkannt wird, wechselt die Markierung von Rot zu Grün.",
    "2. Die zweite Kamera zeigt direkt die Halterung. Sie zeigt, wie und wo genau das entnommene Bauteil platziert werden muss.",
    "",
    "Wenn alles korrekt platziert ist, wird ein Bestätigungston ertönen und der nächste Schritt angezeigt, bis der Unilokk erfolgreich montiert ist.",
    "",
    "Buttons:",
    "- Sprachaufnahme: Drücken Sie diesen Button, um die Sprachaufnahme zu starten. Drücken Sie erneut, um die Aufnahme zu beenden. Die Aufnahme wird transkribiert, was einige Sekunden dauern kann.",
    "- Hilfe (?): Dieser Button zeigt eine Erklärung der GUI an.",
    "- Tagesanalyse: Dieser Button analysiert die Dateien des Tages. Dies kann je nach Anzahl und Größe der Dateien einige Zeit in Anspruch nehmen.",
    "",
    "Folgen Sie bitte den Anweisungen und haben Sie Spaß beim Montieren! :)"
]

for text in info_texts:
    label = Label(info_frame, text=text, font=('Helvetica', 12), fg="#e7e7e7", bg="#17365c", wraplength=550, justify='center')
    label.pack(anchor='center', pady=2)

start_button = tk.Button(welcome_window, text="Start", font=('Helvetica', 14, 'bold'), fg="#17365c", bg="#e7e7e7", command=start_system)
start_button.pack(pady=20)

# Anweisungsinfo-Label
instructions_info_label = tk.Label(root, text="Bitte folgen Sie den Anweisungen:", fg=RUB_GRAU, bg=RUB_BLAU, font=('Helvetica', 14, 'bold'))
instructions_info_label.pack(pady=10)

# Feedback-Label
feedback_label = tk.Label(root, text=steps[current_step_index], fg=RUB_GRUEN, bg=RUB_BLAU, font=('Helvetica', 14, 'bold'), wraplength=window_width - 40, height=3)
feedback_label.pack(pady=(10, 20))  # Mehr vertikalen Platz nach unten hinzufügen

# Anweisungsbild-Label
instruction_label = tk.Label(root, bg=RUB_BLAU)
instruction_label.pack(pady=(0, 20))  # Mehr vertikalen Platz nach unten hinzufügen

# GUI-Komponenten für Teilsystem 2 (Linke Seite)
camera_frame_2 = Frame(root, width=int(window_width * 0.45), height=int(window_height * 0.7), bg=RUB_BLAU)
camera_frame_2.pack(side="left", padx=20, pady=20)
camera_label_2_title = Label(camera_frame_2, text="1. Hier entnehmen:", fg=RUB_GRAU, bg=RUB_BLAU, font=('Helvetica', 14, 'bold'))
camera_label_2_title.pack()
camera_label_2 = Label(camera_frame_2, bg=RUB_BLAU)
camera_label_2.pack()

# GUI-Komponenten für Teilsystem 1 (Rechte Seite)
camera_frame_1 = Frame(root, width=int(window_width * 0.45), height=int(window_height * 0.7), bg=RUB_BLAU)
camera_frame_1.pack(side="right", padx=20, pady=20)
camera_label_1_title = Label(camera_frame_1, text="2. Hier platzieren:", fg=RUB_GRAU, bg=RUB_BLAU, font=('Helvetica', 14, 'bold'))
camera_label_1_title.pack()
camera_label_1 = Label(camera_frame_1, bg=RUB_BLAU)
camera_label_1.pack()

# Threads für die Videoaufnahme starten
threading.Thread(target=capture_loop_1, daemon=True).start()
threading.Thread(target=capture_loop_2, daemon=True).start()

# Labels für den Unilokk-Counter und den Arbeitszeit-Timer
unilokk_counter_label = tk.Label(root, text="Fertige Unilokks: 0", fg=RUB_GRAU, bg=RUB_BLAU, font=('Helvetica', 14, 'bold'))
unilokk_counter_label.pack(pady=(10, 0))

work_time_label = tk.Label(root, text="Arbeitszeit: 00:00:00", fg=RUB_GRAU, bg=RUB_BLAU, font=('Helvetica', 14, 'bold'))
work_time_label.pack(pady=(0, 10))

#Transkription:
record_button = Button(root, text="Starte Sprachaufnahme", font=('Helvetica', 14, 'bold'), fg=RUB_BLAU, bg=RUB_GRAU, command=toggle_recording, width=30)
record_button.pack(pady=20)

recording_status_label = Label(root, text="", font=('Helvetica', 12), fg=RUB_GRUEN, bg=RUB_BLAU)
recording_status_label.pack(pady=10)

# Hilfe-Button
help_button = Button(root, text="?", font=('Helvetica', 14, 'bold'), fg=RUB_BLAU, bg=RUB_GRAU, command=show_help_image, width=3, height=1)
help_button.pack(pady=10)

#Analyse-Button:
analyze_button = Button(root, text="Tagesanalyse", font=('Helvetica', 14, 'bold'), fg=RUB_BLAU, bg=RUB_GRAU, command=analyze_files, width=30)
analyze_button.pack(pady=20)

# Escape-Taste binden, um das Hauptfenster zu schließen
root.bind("<Escape>", close_on_escape)
welcome_window.bind("<Escape>", close_on_escape)

# Kameras aktualisieren
root.protocol("WM_DELETE_WINDOW", on_closing)
root.after(30, update_camera_from_queue, frame_queue_1, camera_label_1)
root.after(30, update_camera_from_queue, frame_queue_2, camera_label_2)
root.mainloop()
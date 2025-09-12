import sqlite3
from datetime import datetime
import streamlit as st
import cv2
import numpy as np
import smtplib
from email.mime.text import MIMEText
import base64
import pandas as pd
from ultralytics import YOLO
import google.generativeai as genai
import PIL.Image
import matplotlib.image as mpimg
from sort import Sort  # Import the SORT tracker
import math
import cvzone
import time

# Initialize the SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Define line limits for counting
count_line = [199, 363, 1208, 377]

# List to store counted IDs
counted_ids = []

#pt.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#reader = easyocr.Reader(['en'], gpu=True)

# Load the best.pt model directly using the ultralytics YOLO class
yolo_plate_model = YOLO(r'best.pt')
yolo_plate_model.to('cpu')#use cpu for processing
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# Initialize SQLite database and create table if not exists
def initialize_database():
    conn = sqlite3.connect('violations.db')
    c = conn.cursor()
    
    # Create table if not exists
    c.execute('''
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_id TEXT NOT NULL,
            violation_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            fine_amount REAL NOT NULL,
            email_sent TEXT NOT NULL DEFAULT 'No',  -- New column to track email status
            speed_kmh REAL DEFAULT 0  -- New column to store speed information
        )
    ''')
    
    # Check if speed_kmh column exists, if not add it
    c.execute("PRAGMA table_info(violations)")
    columns = [column[1] for column in c.fetchall()]
    
    if 'speed_kmh' not in columns:
        c.execute("ALTER TABLE violations ADD COLUMN speed_kmh REAL DEFAULT 0")
        print("Added speed_kmh column to existing violations table")
    
    conn.commit()
    conn.close()

# Call the function to initialize the database
initialize_database()

# Function to update database schema if needed
def update_database_schema():
    conn = sqlite3.connect('violations.db')
    c = conn.cursor()
    
    # Check if speed_kmh column exists, if not add it
    c.execute("PRAGMA table_info(violations)")
    columns = [column[1] for column in c.fetchall()]
    
    if 'speed_kmh' not in columns:
        try:
            c.execute("ALTER TABLE violations ADD COLUMN speed_kmh REAL DEFAULT 0")
            conn.commit()
            print("Successfully added speed_kmh column to violations table")
        except sqlite3.OperationalError as e:
            print(f"Error adding speed_kmh column: {e}")
    
    conn.close()

# Update database schema
update_database_schema()

st.set_page_config(page_title="Traffic Violation Detection System", layout="wide")

def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{b64_encoded});
        background-size: cover;    
   
    }}
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)

set_background("background.png")  #  background image file

st.title("🚦 Traffic Violation Detection System")


# Fine structure based on violation type
FINE_STRUCTURE = {
    "Signal Breaking": 1000,
    "Overspeeding": 1000,
    "Parking Violation": 500,
    "Wrong Lane": 1000,
    "Red Light Violation": 1000,
    
}

# Function to log violations in the database
def log_violation(vehicle_id, violation_type, timestamp, fine_amount, email_sent, speed_kmh=0):
    conn = sqlite3.connect('violations.db')
    c = conn.cursor()
    
    # Check if speed_kmh column exists
    c.execute("PRAGMA table_info(violations)")
    columns = [column[1] for column in c.fetchall()]
    
    if 'speed_kmh' in columns:
        c.execute(
            "INSERT INTO violations (vehicle_id, violation_type, timestamp, fine_amount, email_sent, speed_kmh) VALUES (?, ?, ?, ?, ?, ?)",
            (vehicle_id, violation_type, timestamp, fine_amount, email_sent, speed_kmh)
        )
    else:
        # Fallback for old database without speed_kmh column
        c.execute(
            "INSERT INTO violations (vehicle_id, violation_type, timestamp, fine_amount, email_sent) VALUES (?, ?, ?, ?, ?)",
            (vehicle_id, violation_type, timestamp, fine_amount, email_sent)
        )
    
    conn.commit()
    conn.close()

# Function to send email alerts
def send_email_alert(vehicle_id, violation_type, fine_amount):
    try:  # Email content
        msg = MIMEText(f"""
        <html>
        <body>
            <h2 style="color: red;">Traffic Violation Detected</h2>
            <p><strong>Vehicle ID:</strong> {vehicle_id}</p>
            <p><strong>Violation Type:</strong> {violation_type}</p>
            <p><strong>Fine Amount:</strong> ₹{fine_amount}</p>
            <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Please take the necessary actions to address this violation.</p>
            <hr>
            <h3 style="color: blue;">Traffic Safety Awareness</h3>
            <p>Following traffic rules is essential for your safety and the safety of others on the road. 
            Violations such as <strong>{violation_type}</strong> can lead to accidents and endanger lives. 
            Always adhere to traffic regulations to ensure a safe journey for everyone.</p>
            <p>Thank you for your attention to this matter.</p>
            <p>Best regards,<br>Traffic Violation Detection System</p>
        </body>
        </html>
        """, "html")
        msg['Subject'] = 'Traffic Violation Detected'
        msg['From'] = 'example@gmail.com'  # sender's email
        msg['To'] = 'abc@gmail.com'  # recipient's email
        # SMTP server configuration
        with smtplib.SMTP('smtp.gmail.com', 587) as server:  #  SMTP server
            server.starttls()   # email and app password
            server.login('example@gmail.com', 'your app passwrod') 
            server.send_message(msg)
        print(f"Email sent successfully for Vehicle ID: {vehicle_id}, Violation: {violation_type}")
        return True
    except smtplib.SMTPAuthenticationError:
        print("Failed to send email: Authentication error. Check your email and app password.")
    except smtplib.SMTPConnectError:
        print("Failed to send email: Unable to connect to the SMTP server.")
    except Exception as e:
        print(f"Failed to send email: {e}")
    return False

#def sort_cont(character_contours): 
	
	#To sort contours 
	
	#i = 0
	#boundingBoxes = [cv2.boundingRect(c) for c in character_contours] 
	
	#(character_contours, boundingBoxes) = zip(*sorted(zip(character_contours, 
					#	boundingBoxes), 
				#key = lambda b: b[1][i], 
				#reverse = False)) 
	
#	return character_contours 

#detected_plate_cache = {}
 

def detect_license_plate(vehicle_roi):
    global yolo_plate_model

    # Ensure the ROI has valid dimensions
    if vehicle_roi is None or vehicle_roi.shape[0] == 0 or vehicle_roi.shape[1] == 0:
        return "UNKNOWN"

    # Perform detection using the best.pt model
    results = yolo_plate_model.predict(source=vehicle_roi, save=False, conf=0.5)  # Direct prediction
    detections = results[0].boxes.xyxy.cpu().numpy()  # Extract bounding boxes

    if len(detections) > 0:
        detection = detections[0]  # Use the first detection
        x1, y1, x2, y2 = map(int, detection[:4])
        plate_roi = vehicle_roi[y1:y2, x1:x2]

        # Ensure the plate ROI has valid dimensions
        if plate_roi.shape[0] == 0 or plate_roi.shape[1] == 0:
            return "UNKNOWN"

        # Resize and preprocess the plate ROI
        scaled_height = int(plate_roi.shape[0] * 10)
        scaled_width = int(plate_roi.shape[1] * 10)
        plate_roi_resized = cv2.resize(plate_roi, (scaled_width, scaled_height), interpolation=cv2.INTER_CUBIC)

        # Save and process the image for OCR
        mpimg.imsave(r"plate_roi1.png", plate_roi_resized)
        image = PIL.Image.open(r"plate_roi1.png")
        genai.configure(api_key="AIzaSyB3tzlGcfGzUcGk0nRjOuMKpljb69UYaqE")
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(["Do OCR on it, direct give the output without any other text", image])
        print("Gemini: ", response.text)

        if len(response.text) >= 6:  # Ensure the detected text is of reasonable length
            return response.text  # Return the detected license plate text

    return "UNKNOWN"

# function to check for red light violation
def is_red_light_violation(frame, vehicle_roi):
    """
    Detects if the vehicle is crossing a red light.
    Uses color detection to identify red traffic lights and checks vehicle position.
    """
    # plt.imshow(frame)
    # plt.axis('off')
    # plt.show()  

    # plt.imshow(vehicle_roi)
    # plt.axis('off')     
    # plt.show()

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv_frame, (0, 100, 100), (10, 255, 255))  # Lower red range
    red_mask2 = cv2.inRange(hsv_frame, (160, 100, 100), (180, 255, 255))  # Upper red range
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Check if red light is detected
    if cv2.countNonZero(red_mask) > 1000:  # Threshold for detecting red light
        # Define stop line position (e.g., middle of the frame)
        stop_line_y = frame.shape[0] // 2
        vehicle_position = vehicle_roi.shape[0] // 2

        # Check if the vehicle is past the stop line
        if vehicle_position < stop_line_y:
            return True
    return False

#  function to check for signal breaking
def is_signal_breaking(frame, vehicle_roi):
    """
    Detects if the vehicle crosses the stop line during a red signal.
    """
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv_frame, (0, 100, 100), (10, 255, 255))  # Detect red signal

    # Define stop line position
    stop_line_y = frame.shape[0] // 2
    vehicle_position = vehicle_roi.shape[0] // 2

    # Check if the vehicle crosses the stop line during a red signal
    if cv2.countNonZero(red_mask) > 1000 and vehicle_position < stop_line_y:
        return True
    return False

# function to check for lane violation
def is_lane_violation(frame, vehicle_roi):
    """
    Detects if the vehicle is outside its designated lane.
    Uses edge detection and Hough lines to identify lane markings.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_frame, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # print(f"Line coordinates: {x1}, {y1}, {x2}, {y2}")
            # # Draw the detected lines on the frame (optional)
            # plt.imshow(frame)
            # plt.axis('off')
            # plt.show()
            # cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # plt.imshow(frame)
            # plt.axis('off')
            # plt.show()
            # Check if the vehicle crosses the lane markings
            if x1 < vehicle_roi.shape[1] and x2 > vehicle_roi.shape[1]:
                return True
    return False

# Global variables for speed tracking
vehicle_positions = {}  # Store previous positions of vehicles
vehicle_timestamps = {}  # Store timestamps for speed calculation
speed_limit = 50  # Speed limit in km/h (configurable)

def calculate_speed(vehicle_id, current_position, current_time, pixels_per_meter=20, speed_limit=50):
    """
    Calculate vehicle speed based on position tracking and time elapsed.
    Returns speed in km/h and whether it exceeds the speed limit.
    """
    global vehicle_positions, vehicle_timestamps
    
    if vehicle_id not in vehicle_positions:
        # First detection of this vehicle
        vehicle_positions[vehicle_id] = current_position
        vehicle_timestamps[vehicle_id] = current_time
        return 0, False
    
    # Get previous position and timestamp
    prev_position = vehicle_positions[vehicle_id]
    prev_time = vehicle_timestamps[vehicle_id]
    
    # Calculate time elapsed (in seconds)
    time_elapsed = current_time - prev_time
    
    if time_elapsed <= 0:
        return 0, False
    
    # Calculate distance traveled (Euclidean distance in pixels)
    distance_pixels = np.sqrt((current_position[0] - prev_position[0]) ** 2 +
                             (current_position[1] - prev_position[1]) ** 2)
    
    # Convert pixels to real-world distance
    distance_meters = distance_pixels / pixels_per_meter
    
    # Calculate speed in m/s
    speed_ms = distance_meters / time_elapsed
    
    # Convert to km/h
    speed_kmh = speed_ms * 3.6
    
    # Update stored position and timestamp
    vehicle_positions[vehicle_id] = current_position
    vehicle_timestamps[vehicle_id] = current_time
    
    # Check if exceeding speed limit
    is_overspeeding = speed_kmh > speed_limit
    
    return speed_kmh, is_overspeeding

def is_overspeeding(vehicle_id, current_position, current_time, pixels_per_meter=20, speed_limit=50):
    """
    Check if vehicle is overspeeding based on calculated speed.
    """
    speed, is_overspeeding = calculate_speed(vehicle_id, current_position, current_time, pixels_per_meter, speed_limit)
    return is_overspeeding, speed

#  function to check for parking violation
def is_parking_violation(frame, vehicle_roi):
    """
    Detects if the vehicle is parked in a no-parking zone.
    Uses predefined no-parking zone coordinates.
    """
    # Define no-parking zone (e.g., top-left corner of the frame)
    no_parking_zone = frame[0:100, 0:100]
    vehicle_center = ((vehicle_roi.shape[1] // 2), (vehicle_roi.shape[0] // 2))

    # Check if the vehicle is within the no-parking zone
    if no_parking_zone.shape[0] > vehicle_center[1] > 0 and no_parking_zone.shape[1] > vehicle_center[0] > 0:
        return True
    return False

# Function to detect violations based on vehicle behavior
def detect_violation(vehicle_roi, frame, vehicle_id=None, current_position=None, current_time=None):
    """
    Detects the type of traffic violation based on the vehicle's behavior.
    """
    if is_red_light_violation(frame, vehicle_roi):
        return "Red Light Violation"
    elif is_signal_breaking(frame, vehicle_roi):
        return "Signal Breaking"
    elif is_lane_violation(frame, vehicle_roi):
        return "Wrong Lane"
    elif is_parking_violation(frame, vehicle_roi):
        return "Parking Violation"
    else:
        return None

def detect_speed_violation(vehicle_id, current_position, current_time, pixels_per_meter=20, speed_limit=50):
    """
    Specifically detect speed violations.
    Returns violation type and speed if overspeeding, None otherwise.
    """
    if vehicle_id is not None and current_position is not None and current_time is not None:
        is_overspeeding_flag, speed = is_overspeeding(vehicle_id, current_position, current_time, pixels_per_meter, speed_limit)
        if is_overspeeding_flag:
            return "Overspeeding", speed
    return None, 0

# Function to process video frames
def process_video(video_path, pixels_per_meter=20, speed_limit=50):
    cap = cv2.VideoCapture(video_path)
    detected_vehicles = {}  # Dictionary to map vehicle IDs to license plate numbers
    logged_violations = set()  # Set to track unique violations
     # Counter for vehicles breaking rules
    # Ensure it is accessible across the function
    total_detected_vehicles = 0  # Counter for total vehicles detected
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Setup video writer for output
    output_path = "Resources/output/processed_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1280, 720))

    global model
    if 'model' not in globals():
        model = YOLO('yolov8n.pt')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        results = model.predict(source=frame, save=False, conf=0.5)
        annotated_frame = np.copy(results[0].plot())

        class_labels = [
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", 
            "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", 
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", 
            "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]

        detection_array = np.empty((0, 5))

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width, height = x2 - x1, y2 - y1
                confidence = math.ceil((box.conf[0] * 100)) / 100
                class_id = int(box.cls[0])
                class_name = class_labels[class_id]

                if class_name in ["car", "truck", "motorbike", "bus"] and confidence > 0.3:
                    detection_entry = np.array([x1, y1, x2, y2, confidence])
                    detection_array = np.vstack((detection_array, detection_entry))

        tracked_objects = tracker.update(detection_array)

        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)
            width, height = x2 - x1, y2 - y1

            # Calculate center
            center_x, center_y = x1 + width // 2, y1 + height // 2
            current_position = (center_x, center_y)
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert to seconds

            # --- Counting Detected Vehicles Correctly ---
            if obj_id not in detected_vehicles:
                detected_vehicles[obj_id] = "UNKNOWN"
                total_detected_vehicles += 1

            # --- Check for Violations ---
            vehicle_roi = frame[y1:y2, x1:x2]
            if obj_id in detected_vehicles:
                if detected_vehicles[obj_id] == "UNKNOWN":
                    detected_vehicles[obj_id] = detect_license_plate(vehicle_roi)

                vehicle_id = detected_vehicles[obj_id]
                
                # Check for speed violations
                speed_violation, speed = detect_speed_violation(obj_id, current_position, current_time, pixels_per_meter, speed_limit)
                
                # Check for other violations
                violation_type = detect_violation(vehicle_roi, frame, obj_id, current_position, current_time)
                
                # Use speed violation if detected, otherwise use other violations
                if speed_violation:
                    violation_type = speed_violation

                # Calculate current speed for display
                current_speed, _ = calculate_speed(obj_id, current_position, current_time, pixels_per_meter, speed_limit)

                if violation_type:
                    unique_key = f"{obj_id}_{violation_type}"
                   
                    if unique_key not in logged_violations:
                        logged_violations.add(unique_key)
                       
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        fine_amount = FINE_STRUCTURE.get(violation_type, 500)

                        email_sent = send_email_alert(vehicle_id, violation_type, fine_amount)
                        if email_sent:
                            print(f"Email sent for Vehicle ID: {vehicle_id}, Violation: {violation_type}")
                            
                        # Log the violation in the database 
                        speed_to_log = speed if speed_violation else 0
                        log_violation(vehicle_id, violation_type, timestamp, fine_amount, "Yes" if email_sent else "No", speed_to_log)
                        
                        # Draw red box if violation
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        violation_text = f'{vehicle_id} - {violation_type}'
                        if speed_violation:
                            violation_text += f' ({speed:.1f} km/h)'
                        cv2.putText(annotated_frame, violation_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:
                    # Draw normal bounding box with speed info
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(annotated_frame, f'ID: {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    # Display speed if available
                    if current_speed > 0:
                        speed_text = f'Speed: {current_speed:.1f} km/h'
                        cv2.putText(annotated_frame, speed_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw center point
            cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)
        # Draw count line
        cv2.line(annotated_frame, (count_line[0], count_line[1]), (count_line[2], count_line[3]), (0, 255, 0), 2)

        # Show detection count
        cvzone.putTextRect(annotated_frame, f'COUNT: {len(detected_vehicles)}', (20, 50), scale=1, thickness=2, colorT=(255, 255, 255), colorR=(255, 255, 0), font=cv2.FONT_HERSHEY_SIMPLEX)

        # Write frame to output video
        out.write(annotated_frame)
        
        # Update progress
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        progress = current_frame / frame_count
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {current_frame}/{frame_count} - {progress*100:.1f}%")

    cap.release()
    out.release()

    st.success("Processing done!")
    st.write(f"### Total Vehicles Detected: {total_detected_vehicles}")
    
    # Display the processed video
    st.write("### Processed Video Result:")
    st.video(output_path)
    
    # Add download button for the processed video
    with open(output_path, "rb") as file:
        st.download_button(
            label="Download Processed Video",
            data=file.read(),
            file_name="traffic_violation_processed.mp4",
            mime="video/mp4"
        )
    
# Function to process camera frames in real-time
def process_camera(camera_index=0, pixels_per_meter=20, speed_limit=50):
    cap = cv2.VideoCapture(camera_index)
    detected_vehicles = {}  # Dictionary to map vehicle IDs to license plate numbers
    logged_violations = set()  # Set to track unique violations
    total_detected_vehicles = 0  # Counter for vehicles breaking rules
    stop_camera = False  # Flag to stop the camera

    global model
    if 'model' not in globals():
        model = YOLO('yolov8n.pt')
    stop_camera = st.sidebar.button("Stop Camera")
    while cap.isOpened() and not stop_camera:
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera not available or frame not captured.")
            break

        frame = cv2.resize(frame, (1280, 720))
        results = model.predict(source=frame, save=False, conf=0.5)
        annotated_frame = np.copy(results[0].plot())

        # Collect detections for tracking
        detection_array = np.empty((0, 5))
        for result in results[0].boxes.xyxy.cpu().numpy():
            if len(result) >= 4:
                x1, y1, x2, y2 = map(int, result[:4])
                conf = result[3] if len(result) >= 4 else 0.0

                if conf > 0.5:
                    detection_entry = np.array([x1, y1, x2, y2, conf])
                    detection_array = np.vstack((detection_array, detection_entry))

        # Update tracker with detections
        tracked_objects = tracker.update(detection_array)

        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)
            width, height = x2 - x1, y2 - y1

            # Calculate center of the box
            center_x, center_y = x1 + width // 2, y1 + height // 2
            current_position = (center_x, center_y)
            current_time = time.time()  # Use system time for real-time processing

            # Check if the object crosses the count line
            if count_line[0] < center_x < count_line[2] and count_line[1] - 20 < center_y < count_line[1] + 20:
                if obj_id not in detected_vehicles:
                    detected_vehicles[obj_id] = "UNKNOWN"  # Initialize with UNKNOWN
                    total_detected_vehicles += 1  # Increment total detected vehicles

            # Check for violations
            vehicle_roi = frame[y1:y2, x1:x2]
            if obj_id in detected_vehicles:
                # Detect license plate only if not already detected
                if detected_vehicles[obj_id] == "UNKNOWN":
                    detected_vehicles[obj_id] = detect_license_plate(vehicle_roi)

                vehicle_id = detected_vehicles[obj_id]
                
                # Check for speed violations
                speed_violation, speed = detect_speed_violation(obj_id, current_position, current_time, pixels_per_meter, speed_limit)
                
                # Check for other violations
                violation_type = detect_violation(vehicle_roi, frame, obj_id, current_position, current_time)
                
                # Use speed violation if detected, otherwise use other violations
                if speed_violation:
                    violation_type = speed_violation

                # Calculate current speed for display
                current_speed, _ = calculate_speed(obj_id, current_position, current_time, pixels_per_meter, speed_limit)

                if violation_type:
                    unique_key = f"{obj_id}_{violation_type}"
                    if unique_key not in logged_violations:
                        logged_violations.add(unique_key)
                        total_detected_vehicles += 1  # Increment total violations

                        # Log the violation in the database
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        fine_amount = FINE_STRUCTURE.get(violation_type, 500)
                        speed_to_log = speed if speed_violation else 0
                        log_violation(vehicle_id, violation_type, timestamp, fine_amount, "No", speed_to_log)

                        # Send email alert
                        email_sent = send_email_alert(vehicle_id, violation_type, fine_amount)
                        if email_sent:
                            print(f"Email sent for Vehicle ID: {vehicle_id}, Violation: {violation_type}")

                        # Draw red bounding box for violations
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        violation_text = f'{vehicle_id} - {violation_type}'
                        if speed_violation:
                            violation_text += f' ({speed:.1f} km/h)'
                        cv2.putText(annotated_frame, violation_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:
                    # Draw normal bounding box with speed info
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(annotated_frame, f'ID: {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    # Display speed if available
                    if current_speed > 0:
                        speed_text = f'Speed: {current_speed:.1f} km/h'
                        cv2.putText(annotated_frame, speed_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw center point
            cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)

        # Draw count line
        cv2.line(annotated_frame, (count_line[0], count_line[1]), (count_line[2], count_line[3]), (0, 255, 0), 2)

        # Display count
        cvzone.putTextRect(annotated_frame, f'COUNT: {len(detected_vehicles)}', (20, 50), scale=1, thickness=2, colorT=(255, 255, 255), colorR=(255, 255, 0), font=cv2.FONT_HERSHEY_SIMPLEX)

        st.image(annotated_frame, channels="BGR", use_container_width=True)

        # Add a "Stop Camera" button with a unique key
        
        

    cap.release()

    # Display final counts
    st.success("Camera stream ended.")
    st.write(f"### Total Vehicles Detected: {total_detected_vehicles}")  # Now tracking vehicles
   

# Speed limit configuration
st.sidebar.header("Speed Detection Settings")
speed_limit = st.sidebar.slider("Speed Limit (km/h)", min_value=20, max_value=120, value=50, step=5)
pixels_per_meter = st.sidebar.slider("Pixels per Meter (calibration)", min_value=5, max_value=50, value=20, step=1)

# Start the camera feed or upload a video
option = st.sidebar.selectbox("Choose an option", ["Use Camera", "Upload Video", "View Violations"])

if option == "Use Camera":
    if st.sidebar.button("Start Camera"):
        st.write("Starting the camera...")
        process_camera(pixels_per_meter=pixels_per_meter, speed_limit=speed_limit)

elif option == "Upload Video":
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        st.write("Processing the uploaded video...")
        process_video("temp_video.mp4", pixels_per_meter=pixels_per_meter, speed_limit=speed_limit)

elif option == "View Violations":
    st.header("Violation Records")
    conn = sqlite3.connect('violations.db')
    c = conn.cursor()
    
    # Check if speed_kmh column exists
    c.execute("PRAGMA table_info(violations)")
    columns = [column[1] for column in c.fetchall()]
    
    if 'speed_kmh' in columns:
        c.execute("SELECT id, vehicle_id, violation_type, timestamp, fine_amount, email_sent, speed_kmh FROM violations")
        rows = c.fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=["ID", "Vehicle ID", "Violation Type", "Timestamp", "Fine Amount", "Email Sent", "Speed (km/h)"])
            st.write("### Violation Records Table")
            st.dataframe(df)
            
            # Display speed statistics
            speed_violations = df[df["Violation Type"] == "Overspeeding"]
            if not speed_violations.empty:
                st.write("### Speed Violation Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Speed Violations", len(speed_violations))
                with col2:
                    st.metric("Average Speed", f"{speed_violations['Speed (km/h)'].mean():.1f} km/h")
                with col3:
                    st.metric("Highest Speed", f"{speed_violations['Speed (km/h)'].max():.1f} km/h")
        else:
            st.write("No violations detected.")
    else:
        # Fallback for old database without speed_kmh column
        c.execute("SELECT id, vehicle_id, violation_type, timestamp, fine_amount, email_sent FROM violations")
        rows = c.fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=["ID", "Vehicle ID", "Violation Type", "Timestamp", "Fine Amount", "Email Sent"])
            st.write("### Violation Records Table")
            st.dataframe(df)
            st.info("Speed information not available in this database. Please restart the application to update the database schema.")
        else:
            st.write("No violations detected.")
    
    conn.close()


#!/usr/bin/env python
# coding: utf-8

# # 21MIA1025 - CSE4076 - Lab Assignment 06
Task 1: Tagging a Person in Videos Taken in a Situation
Objective:
Tag and track a person across frames in a video based on appearance using traditional image processing techniques.

Task Description:

1. Load Video:
Load the provided video file using OpenCV.

2. Person Detection:
Use background subtraction or frame differencing to detect moving objects (people) in the video.
Apply basic feature extraction (color histograms, edge features) to isolate the target person based on appearance.

3. Person Tracking:
Implement a tracking algorithm like centroid-based tracking or optical flow to tag and track the detected person across the video frames.

4. Tagging Output:
Visualize the video with a bounding box and label/tag around the identified person as they move through the frames.
# In[1]:


import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(r"F:\7 FALL SEM\Image and Video Analytics\lab 06\task_1.mp4") 

# Get the total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {total_frames}")


# In[2]:


# Initialize the background subtractor (using MOG2)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames_to_display = 5  
frame_start = total_frames - frames_to_display * 30  
# Start capturing from here for the last few frames

# Variables to control frame selection and storage
selected_frames = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction to get the foreground mask
    fg_mask = bg_subtractor.apply(frame)
    
    # Clean up the mask using morphological operations
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over each contour to detect and label people
    for contour in contours:
        if cv2.contourArea(contour) > 500: 
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Person', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Save frames only near the end of the video
    if frame_count >= frame_start and frame_count % 30 == 0 and len(selected_frames) < frames_to_display:
        selected_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    frame_count += 1
    
    cv2.imshow('Detected Persons', frame)

    # Press 'q' to stop the video
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()


# In[3]:


for i, frame in enumerate(selected_frames):
    plt.figure(figsize=(10, 8))  
    plt.imshow(frame)
    plt.axis('off')
    plt.title(f'Selected Frame {i + 1}')
    plt.show()


# In[4]:


# Function for color histogram extraction
def extract_color_histogram(frame, mask=None):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Calculate the histogram of the HSV image
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Function for edge detection using Canny
def extract_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges

# Continue from previous video capture and processing loop
selected_frames_features = []
frame_count = 0

# We will display the last 'frames_to_display' frames
for i, frame in enumerate(selected_frames):
    # Extract color histogram features
    hist = extract_color_histogram(frame)

    # Perform edge detection
    edges = extract_edges(frame)

    # Display the frame with bounding boxes
    plt.figure(figsize=(15, 10))

    # Original frame with bounding boxes
    plt.subplot(1, 2, 1)
    plt.imshow(frame)
    plt.title(f"Frame {i + 1} with Bounding Box")
    plt.axis('off')

    # Edge detection result
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title(f"Frame {i + 1} - Edges")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# In[5]:


plt.plot(hist)
plt.title(f"Frame {i + 1} - Color Histogram")
plt.xlabel("Histogram Bins")
plt.ylabel("Frequency")


# In[6]:


cap = cv2.VideoCapture(r"F:\7 FALL SEM\Image and Video Analytics\lab 06\task_1.mp4")  # Provide your video file path
# Initialize the background subtractor (using MOG2)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

frame_count = 0
frame_interval = 30  # Process every 30th frame

# Create a plot to show the frames in the notebook
plt.figure(figsize=(10, 10))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for optical flow calculation
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply background subtraction to get the foreground mask
    fg_mask = bg_subtractor.apply(frame)
    
    # Clean up the mask using morphological operations
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over each contour to detect and label people
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Adjust threshold for your video
            x, y, w, h = cv2.boundingRect(contour)
            centroid = (int(x + w / 2), int(y + h / 2))  # Calculate the centroid
            
            # Draw bounding box and centroid
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, centroid, 5, (0, 0, 255), -1)  # Red dot for centroid
            cv2.putText(frame, 'Person', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save every nth frame to the plot
    if frame_count % frame_interval == 0:
        # Convert frame to RGB for displaying in Matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Plot the frame in the notebook
        plt.imshow(frame_rgb)
        plt.axis('off')
        plt.title(f'Frame {frame_count}')
        plt.pause(0.01)  # Pause to update the plot
    
    frame_count += 1
    
    # Show the frame in a window (OpenCV window)
    cv2.imshow('Tracking with Centroid', frame)

    # Press 'q' to stop the video
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()


Task 2: Strategic Marketing – Peak Shopping Duration
Objective:
Analyze a video of a shopping area to identify the peak duration when the most people are shopping.

Task Description:

1. Load Video:
Load the surveillance video from a shopping area.

2. People Detection:
Use frame differencing or optical flow to detect motion and identify people entering the frame.
Count the number of people in each frame based on detected regions.

3. Peak Duration Identification:
Calculate the total number of people in the shopping area for each time period (e.g., 10-minute intervals).
Plot the number of people over time and identify the time interval with the highest count of people.

4. Result:
Provide a summary of the peak shopping duration and display the corresponding frames from the video.
# In[7]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

video_path = r"F:\7 FALL SEM\Image and Video Analytics\lab 06\task_2.mp4"
output_folder = "frames"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(video_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {total_frames}")


# In[8]:


# Initialize background subtractor
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Variables to track people count
frame_rate = cap.get(cv2.CAP_PROP_FPS)
interval_duration = 1  # 1 second
frames_per_interval = int(interval_duration * frame_rate)

people_counts = defaultdict(int)  # Stores counts for each interval
interval_index = 0
frame_count = 0
frame_index = 0  # Track total frame index

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Save frame as image
    frame_filename = os.path.join(output_folder, f"frame_{frame_index}.jpg")
    cv2.imwrite(frame_filename, frame)

    # Apply background subtraction
    fg_mask = back_sub.apply(frame)
    
    # Threshold to binary
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    
    # Find contours of moving objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    people_in_frame = 0

    for cnt in contours:
        if cv2.contourArea(cnt) < 1000:  # Ignore small contours
            continue

        # Get bounding box around the person
        x, y, w, h = cv2.boundingRect(cnt)
        people_in_frame += 1

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Count people in current interval
    people_counts[interval_index] += people_in_frame

    # Show frame with bounding boxes
    cv2.imshow("People Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Move to the next interval if necessary
    frame_count += 1
    if frame_count >= frames_per_interval:
        interval_index += 1
        frame_count = 0

    # Move to the next frame
    frame_index += 1

cap.release()
cv2.destroyAllWindows()

# Plot people counts over time intervals
intervals = list(people_counts.keys())
counts = list(people_counts.values())

plt.figure(figsize=(10, 5))
plt.plot(intervals, counts, marker='o', color='b')
plt.title("People Count Over Time")
plt.xlabel("Time Interval (1-second intervals)")
plt.ylabel("People Count")
plt.grid(True)
plt.show()

# Display some of the frames
selected_frames = [cv2.imread(os.path.join(output_folder, f"frame_{i}.jpg")) for i in range(0, frame_index, max(1, frame_index // 10))]
plt.figure(figsize=(50, 50))
for i, frame in enumerate(selected_frames):
    plt.subplot(1, len(selected_frames), i + 1)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f'Frame {i * max(1, frame_index // 10)}')
plt.show()

# Find peak interval
peak_interval = max(people_counts, key=people_counts.get)
print(f"The peak shopping duration is at interval {peak_interval} with {people_counts[peak_interval]} people.")


# In[9]:


video_path = r"F:\7 FALL SEM\Image and Video Analytics\lab 06\task_2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Background subtractor for people detection
fgbg = cv2.createBackgroundSubtractorMOG2()
frame_counts = []

# People detection and counting with bounding boxes
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    person_count = 0
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  # Filter noise
            person_count += 1
            # Draw bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box

    frame_counts.append(person_count)

    # Display the frame with bounding boxes
    cv2.imshow("People Detection with Bounding Boxes", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Plot people counts with frame numbers as x-axis
plt.plot(range(len(frame_counts)), frame_counts)
plt.xlabel("Frame Number")
plt.ylabel("Total People Count")
plt.title("People Count Over Frames in Shopping Area")
plt.show()



# In[10]:


# Identify peak frame for peak shopping duration
peak_frame_index = np.argmax(frame_counts)
print(f"The peak shopping frame is frame number {peak_frame_index} with {frame_counts[peak_frame_index]} people detected.")

# Load the video and set to a position around the peak frame
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, peak_frame_index - 50))  # Start 50 frames before peak for context

# List to store frames with bounding boxes for later display
frames_with_boxes = []

# Process 100 frames around the peak frame
for _ in range(100):
    ret, frame = cap.read()
    if not ret:
        break

    # Detect people and draw bounding boxes
    fgmask = fgbg.apply(frame)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box

    # Convert frame from BGR to RGB for Matplotlib display and store in the list
    frames_with_boxes.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()

# Display the frames using Matplotlib
plt.figure(figsize=(15, 15))
grid_size = int(np.ceil(np.sqrt(len(frames_with_boxes))))
for i, frame in enumerate(frames_with_boxes):
    plt.subplot(grid_size, grid_size, i + 1)
    plt.imshow(frame)
    plt.axis('off')
    plt.title(f'Frame {peak_frame_index - 50 + i}')

plt.tight_layout()
plt.show()

Task 3: Facial Recognition to Check Fraud Cases
Objective:
Identify a suspect by comparing their facial features to a reference image to check for fraud cases, using traditional facial recognition techniques.

Task Description:

1. Load Images and Video:
Load the reference image of the suspect and a video showing multiple faces.

2. Face Detection:
Use Haar Cascades to detect faces in both the reference image and the video.

3. Feature Matching:
Extract facial features using edge detection or geometric facial features (eye spacing, nose length, etc.).
Compare the features of the faces in the video with the reference face to check for a match.

4. Result:
Output the frames where a match is found and highlight the detected face in the video.
# In[11]:


reference_image_path = r"F:\7 FALL SEM\Image and Video Analytics\lab 06\reference_fraud_face.png"
video_path = r"F:\7 FALL SEM\Image and Video Analytics\lab 06\task_3.mp4"
output_folder = r"C:\Users\Abhineswari\Downloads\output_frames"
os.makedirs(output_folder, exist_ok=True)

# Initialize the video capture object
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")


# In[12]:


# displaying the reference image
reference_image = cv2.imread(reference_image_path)
reference_image_rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
plt.imshow(reference_image_rgb)
plt.axis('off')
plt.show()


# In[13]:


import cv2
import os
import matplotlib.pyplot as plt

# Set paths for reference image and video
reference_image_path = r"F:\7 FALL SEM\Image and Video Analytics\lab 06\reference_fraud_face.png"
video_path = r"F:\7 FALL SEM\Image and Video Analytics\lab 06\task_3.mp4"
output_folder = r"C:\Users\Abhineswari\Downloads\output_frames"
os.makedirs(output_folder, exist_ok=True)

# Load the reference image and convert it to grayscale
reference_image = cv2.imread(reference_image_path)
reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect face in the reference image
ref_faces = face_cascade.detectMultiScale(reference_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Ensure there's at least one face detected in the reference image
if len(ref_faces) == 0:
    print("No faces found in the reference image.")
    exit()
else:
    print(f"{len(ref_faces)} face(s) detected in the reference image.")

# Extract the detected face from the reference image (use the largest face)
x, y, w, h = max(ref_faces, key=lambda face: face[2] * face[3])
reference_face = reference_gray[y:y+h, x:x+w]

# Load the video
cap = cv2.VideoCapture(video_path)

# Initialize a frame counter and list to store matched frames with details
frame_count = 0
matched_frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the current frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face in the frame
    for (fx, fy, fw, fh) in faces:
        # Extract the face from the frame
        face_in_frame = gray_frame[fy:fy+fh, fx:fx+fw]

        # Resize the reference face and detected face to the same size for comparison
        resized_reference = cv2.resize(reference_face, (fw, fh))
        match_result = cv2.matchTemplate(face_in_frame, resized_reference, cv2.TM_CCOEFF_NORMED)
        _, match_val, _, _ = cv2.minMaxLoc(match_result)

        # Check if match value exceeds threshold (indicates a match)
        match_threshold = 0.7
        if match_val > match_threshold:
            # Draw a rectangle around the matching face in the frame
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
            cv2.putText(frame, f'Match: {match_val:.2f}', (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"Match found in frame {frame_count} with similarity score: {match_val:.2f}")

            # Save the frame with match to the output folder
            output_frame_path = os.path.join(output_folder, f'frame_{frame_count}.jpg')
            cv2.imwrite(output_frame_path, frame)

            # Append matched frame to list with its number and score
            matched_frames.append((frame.copy(), frame_count, match_val))

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture
cap.release()



# In[14]:


# Display matched frames with bounding boxes, frame numbers, and similarity scores using Matplotlib
if matched_frames:
    plt.figure(figsize=(15, 15))
    num_frames = len(matched_frames)
    grid_size = int(num_frames ** 0.5) + 1  # Create a square-like grid

    for idx, (matched_frame, frame_num, score) in enumerate(matched_frames):
        plt.subplot(grid_size, grid_size, idx + 1)
        plt.imshow(cv2.cvtColor(matched_frame, cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {frame_num}\nScore: {score:.2f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
else:
    print("No matching frames found.")

Task 4: Number of People Entering and Exiting the Shop
Objective:
Count the number of people entering and exiting a shop based on video footage, using basic motion detection techniques.

Task Description:

1. Load Video:
Load the provided surveillance video of the shop entrance.

2. Motion Detection:
Use frame differencing or optical flow to detect motion as people enter and exit the shop.
Define a region of interest (ROI) near the entrance to focus on counting people.

3. Counting People:
Track the direction of motion (inward or outward) based on detected motion in the ROI.
Increment a counter for each person entering and exiting.

4. Result:
Display the total number of people who entered and exited the shop during the recorded period.

# In[15]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the video
video_path = r"F:\7 FALL SEM\Image and Video Analytics\lab 06\task_4.mp4"
cap = cv2.VideoCapture(video_path)

# Define region of interest (ROI) for entrance counting
roi_top_left = (200, 100)  # Adjust these coordinates as needed
roi_bottom_right = (300, 300)  # Adjust these coordinates as needed

# Background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

# Counters for people entering and exiting
enter_count = 0
exit_count = 0

# Lists to store frames where people entered and exited
entered_frames = []
exited_frames = []

# Previous centroids to track direction
previous_centroids = []

# Function to determine direction of movement
def get_direction(new_centroid, old_centroid):
    if new_centroid[1] < old_centroid[1]:
        return "enter"
    elif new_centroid[1] > old_centroid[1]:
        return "exit"
    return None

# Frame counter for keeping track of frame number
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Define the region of interest in the frame
    roi = frame[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]

    # Apply the background subtractor to detect motion in the ROI
    fgmask = fgbg.apply(roi)
    _, thresh = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  # Filter small contours to avoid noise
            # Get the bounding box and draw it
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate centroid of the bounding box
            centroid = (x + w // 2, y + h // 2)
            centroids.append(centroid)

            # Draw the centroid
            cv2.circle(roi, centroid, 5, (0, 0, 255), -1)

    # Compare centroids with previous centroids to detect direction
    for new_centroid in centroids:
        for old_centroid in previous_centroids:
            direction = get_direction(new_centroid, old_centroid)
            if direction == "enter":
                enter_count += 1
                entered_frames.append(frame.copy())
                print(f"Person Entered in frame {frame_count}")
            elif direction == "exit":
                exit_count += 1
                exited_frames.append(frame.copy())
                print(f"Person Exited in frame {frame_count}")

    # Update previous centroids for the next frame
    previous_centroids = centroids

    # Frame counter update
    frame_count += 1

cap.release()

# Final result
print(f"Total people entered: {enter_count}")
print(f"Total people exited: {exit_count}")



# In[16]:


# Display entered and exited frames using matplotlib
def display_frames(frames, title):
    fig, axes = plt.subplots(1, min(5, len(frames)), figsize=(20, 10))
    fig.suptitle(title)
    for i, ax in enumerate(axes):
        if i < len(frames):
            ax.imshow(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
            ax.set_title(f"Frame {i+1}")
            ax.axis("off")
    plt.show()

# Show frames where people entered
display_frames(entered_frames, "Frames with People Entering")

# Show frames where people exited
display_frames(exited_frames, "Frames with People Exiting")

Task 5: Dwelling Time in a Shopping Mall
Objective:
Measure the amount of time a person or object dwells in a certain area of the shopping mall using video footage.

Task Description:

Load Video:

Load the surveillance video of the shopping mall.
Object/Person Detection:

Use background subtraction or motion detection to detect and track objects or people in the video.
Dwelling Time Calculation:

Set a region of interest (ROI) in the video representing a specific area of the mall.
Track the time each detected person/object spends in the ROI.
Result:

Display the total dwelling time for each person/object detected in the ROI.
# In[3]:


import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=700, varThreshold=50, detectShadows=False)

# Load the video
video_path = r"F:\7 FALL SEM\Image and Video Analytics\lab 06\task_5.mp4" # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)

# Frames per second of the video to calculate time per frame
fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1 / fps  # Time per frame in seconds

# Dictionary to store dwell time for each unique object
dwell_times = defaultdict(float)
object_centers = {}  # Dictionary to store the last known position of each object
object_id_count = 0  # Counter to assign unique IDs to each object

# List to store frames for Matplotlib
captured_frames = []
max_frames_to_capture = 5  # Number of frames to capture and display
frame_counter = 0  # Keep track of frames processed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Additional filtering to clean up the foreground mask
    fgmask = cv2.medianBlur(fgmask, 5)

    # Threshold to create binary image
    _, thresh = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_frame_centers = []  # Track the current frame's object centers

    for cnt in contours:
        # Filter out small contours
        if cv2.contourArea(cnt) < 1000:
            continue

        # Get bounding box for each object
        x, y, w, h = cv2.boundingRect(cnt)
        center = (x + w // 2, y + h // 2)

        # Check if the object matches any previous object based on proximity
        matched_object_id = None
        for obj_id, prev_center in object_centers.items():
            if np.linalg.norm(np.array(center) - np.array(prev_center)) < 50:
                matched_object_id = obj_id
                break

        # If no match found, assign a new ID
        if matched_object_id is None:
            matched_object_id = object_id_count
            object_id_count += 1

        # Update the center position of the matched object
        object_centers[matched_object_id] = center
        current_frame_centers.append(matched_object_id)

        # Increment the dwell time for this object by the time per frame
        dwell_times[matched_object_id] += frame_time

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {matched_object_id} Time: {dwell_times[matched_object_id]:.2f}s", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Remove objects that are no longer detected in the frame
    object_centers = {obj_id: center for obj_id, center in object_centers.items() if obj_id in current_frame_centers}

    # Capture a random frame if we haven’t reached the max number yet
    if len(captured_frames) < max_frames_to_capture and random.random() < 0.01:  # 1% chance to capture frame
        captured_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Display the frame with detections
    cv2.imshow("Dwell Time Detection", frame)

    # Exit condition
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    frame_counter += 1

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Print the total dwell time for each object/person detected in the video
print("Total Dwell Time for Each Detected Object:")
for obj_id, time in dwell_times.items():
    print(f"Object ID {obj_id}: {time:.2f} seconds")

# Display captured frames with bounding boxes using Matplotlib
plt.figure(figsize=(10, 10))
for i, img in enumerate(captured_frames):
    plt.subplot(2, len(captured_frames), i + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Frame {i+1}")
plt.tight_layout()
plt.show()

Task 6: Spotting and Counting a Branded Car in a Video
Objective:
Identify and count the number of branded cars (e.g., a specific logo or color) in a video sequence using feature-based matching.

Task Description:

1. Load Video:
Load the provided video showing vehicles.

2. Car Detection:
Use background subtraction or motion detection to detect moving cars in the video.

3. Feature Matching:
Use color-based detection or template matching to identify the specific branded car (e.g., a car with a specific color or logo).
Track the occurrence of this branded car across the video frames.

4. Counting:
Count the number of times the branded car appears in the video.

5. Result:
Output the total count and display the frames where the branded car is detected.
# In[2]:


import cv2
import numpy as np

# Path to the uploaded video
video_path = r"F:\7 FALL SEM\Image and Video Analytics\lab 06\task_6.mp4"  # Replace with the path to your video file

# Load the video
cap = cv2.VideoCapture(video_path)

# Check if video loaded successfully
if not cap.isOpened():
    raise Exception("Error: Could not open video file.")

# Background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2()

# Define refined color ranges for red cars in HSV color space
color_ranges = {
    "red1": ((0, 70, 50), (10, 255, 255)),  # First range for red
    "red2": ((160, 70, 50), (180, 255, 255)),  # Second range for red
}

# Initialize red car count and list to store frames where red cars are detected
red_car_count = 0
detected_red_frames = []

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply background subtraction to detect moving objects
    fgmask = fgbg.apply(frame)
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    
    # Find contours of the detected areas
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and process each contour
    for cnt in contours:
        if cv2.contourArea(cnt) < 500:  # Minimum area to filter noise
            continue
        
        # Get bounding box for each detected car
        x, y, w, h = cv2.boundingRect(cnt)
        car_roi = frame[y:y+h, x:x+w]  # Region of interest (ROI) for the car
        
        # Convert ROI to HSV color space
        hsv_roi = cv2.cvtColor(car_roi, cv2.COLOR_BGR2HSV)
        
        # Check for red car color match
        for color, (lower, upper) in color_ranges.items():
            if color == "red2":
                # Second range for red color (as red wraps in HSV)
                mask = cv2.inRange(hsv_roi, color_ranges["red1"][0], color_ranges["red1"][1]) | \
                       cv2.inRange(hsv_roi, color_ranges["red2"][0], color_ranges["red2"][1])
            else:
                mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
            
            # Check if a sufficient area within the ROI matches the color
            if cv2.countNonZero(mask) > (0.2 * w * h):  # At least 20% of the area matches
                # Increment red car count and store the frame where it was detected
                red_car_count += 1
                detected_red_frames.append(frame.copy())  # Store the frame with the red car detection
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
                cv2.putText(frame, f"Red Car Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                break  # Stop checking further colors once a match is found for red

    # Display the frame with detections
    cv2.imshow('Red Car Detection', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Print the total count of red branded cars detected
print(f"Total red branded car detections: {red_car_count}")


# In[19]:


if detected_red_frames:
    print(f"Displaying {min(len(detected_red_frames), 20)} initial frames where red cars were detected.")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(30, 30))
    
    # Display the frames in 4 rows and 5 columns
    for i in range(min(len(detected_red_frames), 20)):
        row = i // 5  # Calculate row index (0-3)
        col = i % 5   # Calculate column index (0-4)
        plt.subplot(4, 5, i + 1)  # 4 rows, 5 columns
        plt.imshow(cv2.cvtColor(detected_red_frames[i], cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"Detection {i + 1}")
    
    plt.tight_layout()
    plt.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# 21MIA1025 - CSE4076 - LAB ASSIGNMENT 05

# # Task 1: Motion Estimation and Event Detection in a Video

# Step 1: Load the video using opencv

# In[1]:


import os
from glob import glob
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import cv2
import os

# Load the video
video_path = "C:\\Users\\Abhineswari\\Downloads\\Doggy.mp4"  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize variables
frames = []
frame_count = 0
output_folder = "C:\\Users\\Abhineswari\\Downloads\\OutputFrames"  # Replace with your desired output folder

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read frames from the video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
    frame_count += 1
    
    # Save the frame as an image file
    frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")  # Zero-padded filenames
    cv2.imwrite(frame_filename, frame)

# Release the video capture object
cap.release()

print(f"Total number of frames: {frame_count}")
print(f"Frames saved to: {output_folder}")


# Step 2: Motion Estimation:
# 
# Use frame differencing (Histogram comparision) to detect changes between consecutive frames.
# Subtract consecutive frames and threshold the difference to identify regions of motion.

# In[3]:


# Load the video
video_path = "C:\\Users\\Abhineswari\\Downloads\\Doggy.mp4"  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize variables
frames = []
frame_count = 0
output_folder = "C:\\Users\\Abhineswari\\Downloads\\OutputFrames"  # Replace with your desired output folder
motion_folder = "C:\\Users\\Abhineswari\\Downloads\\MotionEstimation"  # New folder for motion images

# Create output folders if they don't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(motion_folder, exist_ok=True)

# Read frames from the video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
    frame_count += 1
    
    # Save the frame as an image file
    frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)

# Release the video capture object
cap.release()

print(f"Total number of frames: {frame_count}")
print(f"Frames saved to: {output_folder}")

# Motion detection using frame differencing
for i in range(1, frame_count):
    # Calculate the absolute difference between consecutive frames
    frame1 = frames[i - 1]
    frame2 = frames[i]
    diff = cv2.absdiff(frame1, frame2)
    
    # Convert the difference image to grayscale
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get a binary image
    _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    
    # Save the motion detection image
    motion_filename = os.path.join(motion_folder, f"motion_{i:04d}.jpg")
    cv2.imwrite(motion_filename, thresh)

print(f"Motion detection images saved to: {motion_folder}")


# In[4]:


# Function to calculate histogram
def calculate_histogram(image):
    histogram = np.zeros(256)
    for value in image.flatten():
        histogram[value] += 1
    return histogram

# Function to compare histograms using correlation
def compare_histograms(hist1, hist2):
    # Calculate the correlation coefficient
    return np.corrcoef(hist1, hist2)[0, 1]

# Step 3: Histogram Comparison
histogram_scores = []

for i in range(1, len(frames)):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

    # Calculate histograms
    hist1 = calculate_histogram(gray1)
    hist2 = calculate_histogram(gray2)

    # Normalize histograms
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)

    # Compare histograms
    score = compare_histograms(hist1, hist2)
    histogram_scores.append(score)

    # Print the comparison score
    print(f"Histogram comparison score for frame {i-1} and frame {i}: {score:.4f}")


# In[5]:


# Plot scores
plt.plot(range(1, len(histogram_scores) + 1), histogram_scores)
plt.xlabel('Frame Pairs')
plt.ylabel('Histogram Comparison Score')
plt.title('Histogram Comparison Scores between Consecutive Frames')
plt.show()


# In[6]:


# Define the output folder where frames are saved
output_folder = "C:\\Users\\Abhineswari\\Downloads\\OutputFrames"  # Replace with your output folder

# List all frame image files in the output folder
image_paths = sorted([os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.jpg')])

# Select indices for the frames you want to compare
idx = 56  # You can change this index as needed

if idx < len(image_paths) - 1:
    # Read the images from the output folder
    img1_rgb = cv2.cvtColor(cv2.imread(image_paths[idx]), cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(cv2.imread(image_paths[idx + 1]), cv2.COLOR_BGR2RGB)

    # Convert to grayscale
    img1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)

    # Compute grayscale image difference
    grayscale_diff = cv2.subtract(img2, img1)

    # Display the images and their difference
    fig, ax = plt.subplots(1, 3, figsize=(25, 25))
    ax[0].imshow(img1_rgb)
    ax[0].set_title('Frame 1')
    ax[0].axis('off')  # Hide axis
    ax[1].imshow(img2_rgb)
    ax[1].set_title('Frame 2')
    ax[1].axis('off')  # Hide axis
    ax[2].imshow(grayscale_diff * 50, cmap='gray')  # Scale the frame difference to show the noise
    ax[2].set_title('Frame Difference')
    ax[2].axis('off')  # Hide axis
    plt.show()
else:
    print("Index out of bounds for frame extraction.")


# In[7]:


def get_mask(frame1, frame2, kernel=np.array((9, 9), dtype=np.uint8)):
    """Obtains image mask.
    
    Inputs: 
        frame1 - Grayscale frame at time t
        frame2 - Grayscale frame at time t + 1
        kernel - (NxN) array for Morphological Operations
    Outputs: 
        mask - Thresholded mask for moving pixels
    """
    frame_diff = cv2.subtract(frame2, frame1)

    # Blur the frame difference
    frame_diff = cv2.medianBlur(frame_diff, 3)
    
    mask = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 3)

    mask = cv2.medianBlur(mask, 3)

    # Morphological operations
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask

# Define the output folder where frames are saved
output_folder = "C:\\Users\\Abhineswari\\Downloads\\OutputFrames"  # Replace with your output folder

# Select the range of frames to compare (from frame_0001 to frame_0269)
start_index = 1
end_index = 269

# Loop through the selected frames
for idx in range(start_index, end_index):
    frame1_path = os.path.join(output_folder, f"frame_{idx:04d}.jpg")
    frame2_path = os.path.join(output_folder, f"frame_{idx + 1:04d}.jpg")

    if os.path.exists(frame1_path) and os.path.exists(frame2_path):
        # Read the images
        img1_rgb = cv2.cvtColor(cv2.imread(frame1_path), cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(cv2.imread(frame2_path), cv2.COLOR_BGR2RGB)

        # Convert to grayscale
        img1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)

        # Obtain the mask for the moving pixels
        mask = get_mask(img1, img2)

        # Display the images and their mask
        fig, ax = plt.subplots(1, 3, figsize=(25, 25))
        ax[0].imshow(img1_rgb)
        ax[0].set_title(f'Frame {idx:04d}')
        ax[0].axis('off')  # Hide axis
        ax[1].imshow(img2_rgb)
        ax[1].set_title(f'Frame {idx + 1:04d}')
        ax[1].axis('off')  # Hide axis
        ax[2].imshow(mask, cmap='gray')  # Show the mask
        ax[2].set_title('Motion Mask')
        ax[2].axis('off')  # Hide axis
        plt.show()
    else:
        print(f"One of the frames does not exist: {frame1_path} or {frame2_path}")


# Step 3: Event Detection:
# 
# Detect significant motion events, such as sudden movements or object appearances, based on the intensity of motion detected in specific regions.
# Identify and mark frames where events occur based on changes in motion intensity or region activity.
# 

# In[8]:


def get_mask(frame1, frame2, kernel=np.array((9, 9), dtype=np.uint8)):
    """Obtains image mask.
    
    Inputs: 
        frame1 - Grayscale frame at time t
        frame2 - Grayscale frame at time t + 1
        kernel - (NxN) array for Morphological Operations
    Outputs: 
        mask - Thresholded mask for moving pixels
    """
    frame_diff = cv2.subtract(frame2, frame1)
    frame_diff = cv2.medianBlur(frame_diff, 3)
    
    mask = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 3)

    mask = cv2.medianBlur(mask, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask

def get_contour_detections(mask, thresh=400):
    """Obtains initial proposed detections from contours discovered on the mask.
    
    Inputs:
        mask - thresholded image mask
        thresh - threshold for contour size
    Outputs:
        detections - array of proposed detection bounding boxes and scores [[x1,y1,x2,y2,s]]
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > thresh:  # Hyperparameter
            detections.append([x, y, x + w, y + h, area])

    return np.array(detections)

# Define the output folder where frames are saved
output_folder = "C:\\Users\\Abhineswari\\Downloads\\OutputFrames"  # Replace with your output folder

# Select the range of frames to compare (from frame_0001 to frame_0269)
start_index = 1
end_index = 269

# Loop through the selected frames
for idx in range(start_index, end_index):
    frame1_path = os.path.join(output_folder, f"frame_{idx:04d}.jpg")
    frame2_path = os.path.join(output_folder, f"frame_{idx + 1:04d}.jpg")

    if os.path.exists(frame1_path) and os.path.exists(frame2_path):
        # Read the images
        img1_rgb = cv2.cvtColor(cv2.imread(frame1_path), cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(cv2.imread(frame2_path), cv2.COLOR_BGR2RGB)

        # Convert to grayscale
        img1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)

        # Obtain the mask for the moving pixels
        mask = get_mask(img1, img2)

        # Obtain contour detections
        detections = get_contour_detections(mask, thresh=400)

        # Draw bounding boxes on the mask
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        bboxes = detections[:, :4]
        
        for box in bboxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(mask_rgb, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Display the mask with detections
        plt.imshow(mask_rgb)
        plt.title(f"Detected Movers between Frame {idx:04d} and Frame {idx + 1:04d}")
        plt.axis('off')
        plt.show()
    else:
        print(f"One of the frames does not exist: {frame1_path} or {frame2_path}")


# Step 4: Result:
# 
# Visualize motion by highlighting moving regions in each frame.
# Display and annotate the frames where events were detected, along with timestamps.

# In[9]:


def get_mask(frame1, frame2, kernel=np.array((9, 9), dtype=np.uint8)):
    """Obtains image mask."""
    frame_diff = cv2.subtract(frame2, frame1)
    frame_diff = cv2.medianBlur(frame_diff, 3)
    
    mask = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 3)

    mask = cv2.medianBlur(mask, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask

def get_contour_detections(mask, thresh=400):
    """Obtains initial proposed detections from contours discovered on the mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > thresh:  # Hyperparameter
            detections.append([x, y, x + w, y + h, area])

    return np.array(detections)

# Define the output folder where frames are saved
output_folder = "C:\\Users\\Abhineswari\\Downloads\\OutputFrames"  # Replace with your output folder

# Select the range of frames to compare (from frame_0001 to frame_0269)
start_index = 1
end_index = 269

# Loop through the selected frames
for idx in range(start_index, end_index):
    frame1_path = os.path.join(output_folder, f"frame_{idx:04d}.jpg")
    frame2_path = os.path.join(output_folder, f"frame_{idx + 1:04d}.jpg")

    if os.path.exists(frame1_path) and os.path.exists(frame2_path):
        # Read the images
        img1_rgb = cv2.cvtColor(cv2.imread(frame1_path), cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(cv2.imread(frame2_path), cv2.COLOR_BGR2RGB)

        # Convert to grayscale
        img1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)

        # Obtain the mask for the moving pixels
        mask = get_mask(img1, img2)

        # Obtain contour detections
        detections = get_contour_detections(mask, thresh=400)

        # If detections exist, highlight moving regions
        if detections.shape[0] > 0:
            # Draw bounding boxes on the first frame
            for box in detections[:, :4]:
                x1, y1, x2, y2 = box
                cv2.rectangle(img1_rgb, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # Add timestamp to the frame
            timestamp = f"Frame: {idx}, Time: {idx / 30:.2f} sec"  # Assuming 30 FPS
            cv2.putText(img1_rgb, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Print the timestamp and frame number
            print(f"Detected motion in Frame {idx}: Time {idx / 30:.2f} sec")

            # Display the frame with detections
            plt.imshow(img1_rgb)
            plt.title("Detected Movers")
            plt.axis('off')
            plt.show()

    else:
        print(f"One of the frames does not exist: {frame1_path} or {frame2_path}")


# # Task 2: Estimating Sentiments of People in a Crowd – Gesture Analysis and Image Categorization
# 
# Objective:
# Estimate the sentiments of individuals in a crowd using basic gesture analysis techniques, such as detecting facial expressions or hand gestures, without using machine learning models.

# Step 1: Load Image Set:
# 
# Load the provided images of people in a crowd.

# In[64]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread(r"C:\Users\Abhineswari\Downloads\group.jpg")

# Check if the image is loaded properly
if image is not None:
    print("Image loaded successfully!")
else:
    print("Error loading image.")

# Convert to RGB for display using matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image
plt.imshow(image_rgb)
plt.title('Crowd Image')
plt.axis('off')
plt.show()



# Step 2: Preprocessing
# We need to perform face detection and hand gesture detection using tradition techniques.
# 
# Face Detection using Skin-Color-Based Detection
# 
# We can use a color threshold in the HSV (Hue, Saturation, Value) color space to detect skin tones.
# 
# A simple approach for detecting hand gestures would involve detecting contours that match the expected shape of hands or using similar thresholding.

# In[65]:


def skin_color_detection(image):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask using the bounds
    mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    # Apply the mask to extract skin regions
    skin = cv2.bitwise_and(image, image, mask=mask)

    return skin

# Apply skin detection
skin_regions = skin_color_detection(image)

# Display detected skin regions
plt.imshow(cv2.cvtColor(skin_regions, cv2.COLOR_BGR2RGB))
plt.title('Detected Skin Regions')
plt.axis('off')
plt.show()


# Step 3: Gesture Analysis:
# 
# Perform facial feature extraction using geometric methods (e.g., detecting the positions of the eyes, mouth, and eyebrows).
# Classify basic emotions (happy, sad, neutral) based on facial geometry:
# Smiling (upward curvature of the mouth) could indicate happiness.
# A frowning face (downward curvature of the mouth, raised eyebrows) could indicate sadness.
# 
# -> We will modify the code to detect multiple faces, extract features (eyes and mouth) for each face, and classify the emotion of each individual.

# In[66]:


eye_cascade = cv2.CascadeClassifier(r"C:\Users\Abhineswari\Downloads\haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier(r"C:\Users\Abhineswari\Downloads\haarcascade_smile.xml")
face_cascade = cv2.CascadeClassifier(r"C:\Users\Abhineswari\Downloads\haarcascade_frontalface_default.xml")
# Check if cascades are loaded
if eye_cascade.empty():
    print("Error loading eye cascade")
if face_cascade.empty():
    print("Error loading nose cascade")
if mouth_cascade.empty():
    print("Error loading mouth cascade")


# In[67]:


def detect_faces_and_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 2)
    
    detected_faces = []
    for (x, y, w, h) in faces:
        # Draw rectangle around each face
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Focus on face region for detecting eyes and mouth
        face_region = gray[y:y+h, x:x+w]
        color_face_region = image[y:y+h, x:x+w]
        
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(face_region)
        # Detect mouth (smile)
        mouth = mouth_cascade.detectMultiScale(face_region, 1.7, 22)

        # Append face features (eyes and mouth) for later emotion classification
        detected_faces.append((eyes, mouth))

    return detected_faces

# Detect all faces and their features (eyes and mouth)
faces_features = detect_faces_and_features(image)

# Print how many faces were detected
print(f"Number of faces detected: {len(faces_features)}")

# Display the image with rectangles around detected faces
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Faces Detected')
plt.axis('off')
plt.show()


# -> For each detected face, we classify emotions based on the presence of a smile (mouth) and print the result for each individual.

# In[74]:


def classify_emotion(eyes, mouth):
    # Classify as happy if mouth width is large (indicating a smile)
    if len(mouth) > 0:
        (mx, my, mw, mh) = mouth[0]
        mouth_aspect_ratio = mw / mh
        
        if mouth_aspect_ratio > 1.5:  # Larger aspect ratio means smile
            print("Smile detected! Classifying as happy.")
            return 'happy'
        elif mouth_aspect_ratio < 0.8:  # Smaller ratio, frown-like mouth
            print("Downward curve detected. Classifying as sad.")
            return 'sad'
        else:
            print("Neutral or slight curve detected. Classifying as annoyed.")
            return 'annoyed'
    else:
        # No mouth detected; assume neutral or annoyed based on eyes
        print("No smile detected. Checking eyes for other emotions.")
        if len(eyes) > 0:
            # Simple heuristic: smaller eye aspect ratio can indicate annoyance (narrowed eyes)
            (ex, ey, ew, eh) = eyes[0]
            eye_aspect_ratio = ew / eh
            if eye_aspect_ratio < 0.8:  # Smaller eye aspect ratio (narrow eyes)
                print("Narrow eyes detected. Classifying as annoyed.")
                return 'annoyed'
            else:
                print("Eyes wide open. Classifying as neutral.")
                return 'neutral'
        else:
            print("No eyes detected. Defaulting to neutral.")
            return 'neutral'


# Step 4: Image Categorization:
# 
# Categorize the images based on the overall sentiment detected by averaging the identified sentiments of individuals in the crowd (e.g., majority happy, majority sad).

# In[75]:


# Classify emotions for all detected faces using refined rules
emotions = []
for eyes, mouth in faces_features:
    emotion = classify_emotion(eyes, mouth)
    emotions.append(emotion)

# Print detected emotions for each person
for i, emotion in enumerate(emotions, 1):
    print(f"Person {i}: {emotion}")


# Step 5: Result:
# 
# Output the sentiment of each individual and the overall sentiment of the crowd.
# Display the key facial features used for gesture analysis.

# In[76]:


def categorize_crowd(emotions):
    # Count each emotion
    happy_count = emotions.count('happy')
    sad_count = emotions.count('sad')
    annoyed_count = emotions.count('annoyed')
    neutral_count = emotions.count('neutral')

    # Determine majority sentiment based on counts
    if happy_count > sad_count and happy_count > annoyed_count and happy_count > neutral_count:
        print("Majority of the crowd is happy.")
        return 'Majority Happy'
    elif sad_count > happy_count and sad_count > annoyed_count and sad_count > neutral_count:
        print("Majority of the crowd is sad.")
        return 'Majority Sad'
    elif happy_count > happy_count and annoyed_count > sad_count and annoyed_count > neutral_count:
        print("Majority of the crowd is annoyed.")
        return 'Majority Annoyed'
    elif neutral_count > happy_count and neutral_count > sad_count and neutral_count > annoyed_count:
        print("Majority of the crowd is neutral.")
        return 'Majority Neutral'
    else:
        # Handle ties by printing that the sentiment is mixed
        print("Crowd sentiment is mixed.")
        return 'Mixed Sentiment'

# Example output based on the provided counts
overall_sentiment = categorize_crowd(emotions)
print(f'Overall crowd sentiment: {overall_sentiment}')


# In[77]:


# Display the final image with annotations
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title(f'Crowd Sentiment: {overall_sentiment}')
plt.axis('off')
plt.show()


# # Task 3: Gender Identification from Facial Features
# 
# Objective:
# Identify the gender of individuals based on facial features using traditional image processing and feature extraction techniques without using machine learning models.

# Step 1: Load Dataset:
# 
# Load the facial image dataset labeled with gender.

# In[26]:


import cv2
import os
import numpy as np

# Set paths for training data
train_path = r"F:\7 FALL SEM\Image and Video Analytics\lab 05\training"
validation_path = r"F:\7 FALL SEM\Image and Video Analytics\lab 05\validation"

# Get list of male and female directories
male_train_dir = os.path.join(train_path, 'male')
female_train_dir = os.path.join(train_path, 'female')

# Read image filenames
male_images = [os.path.join(male_train_dir, img) for img in os.listdir(male_train_dir) if img.endswith('.jpg')]
female_images = [os.path.join(female_train_dir, img) for img in os.listdir(female_train_dir) if img.endswith('.jpg')]

# Define a function to read and resize images
def load_images(image_paths, size=(128, 128)):
    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.resize(img, size)
        images.append(img)
    return np.array(images)

# Load training images
male_train_images = load_images(male_images)
female_train_images = load_images(female_images)

print(f'Male images: {len(male_train_images)}, Female images: {len(female_train_images)}')


# Step 2: Preprocessing:
# 
# Detect faces in the images using a technique like Haar Cascades.
# Normalize and crop the facial regions for feature extraction.

# In[33]:


import random

# Load the pre-trained Haar Cascade classifier for face detection
cascade_path = r"C:\Users\Abhineswari\Downloads\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Define a function to detect faces in the images
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(30, 30))
    return faces

# Select 5 random images from the male_train_images list
random_images = random.sample(list(male_train_images), 5)

# Loop through the random images and process each one
for i, image in enumerate(random_images):
    # Detect faces in the image
    faces_detected = detect_faces(image)
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces_detected:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Convert the image from BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the image using matplotlib
    plt.subplot(1, 5, i+1)  # Display 5 images in a row
    plt.imshow(image_rgb)
    plt.axis('off')  # Turn off axis labels

# Show all images at once
plt.show()


# In[34]:


# Define a function to crop and normalize faces
def crop_and_normalize_faces(images):
    cropped_faces = []
    for img in images:
        faces = detect_faces(img)
        for (x, y, w, h) in faces:
            # Crop the face region
            face_region = img[y:y+h, x:x+w]
            # Resize to a standard size
            face_resized = cv2.resize(face_region, (128, 128))
            cropped_faces.append(face_resized)
    return np.array(cropped_faces)

# Apply the function to both male and female images
male_faces = crop_and_normalize_faces(male_train_images)
female_faces = crop_and_normalize_faces(female_train_images)

# Check how many faces were detected and cropped
print(f'Male faces detected: {len(male_faces)}, Female faces detected: {len(female_faces)}')


# In[35]:


eye_cascade = cv2.CascadeClassifier(r"C:\Users\Abhineswari\Downloads\haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier(r"C:\Users\Abhineswari\Downloads\haarcascade_mcs_mouth.xml")
nose_cascade = cv2.CascadeClassifier(r"C:\Users\Abhineswari\Downloads\haarcascade_mcs_nose.xml")

# Check if cascades are loaded
if eye_cascade.empty():
    print("Error loading eye cascade")
if nose_cascade.empty():
    print("Error loading nose cascade")
if mouth_cascade.empty():
    print("Error loading mouth cascade")



# In[37]:


# Define a function to extract geometric features and annotate them
def extract_geometric_features_and_display(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect eyes, nose, and mouth
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    nose = nose_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    mouth = mouth_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    features = {}
    
    # Draw eyes and calculate distance between them
    if len(eyes) >= 2:
        eye1, eye2 = eyes[0], eyes[1]
        eye_distance = np.linalg.norm(np.array([eye1[0], eye1[1]]) - np.array([eye2[0], eye2[1]]))
        features['eye_distance'] = eye_distance
        cv2.rectangle(image, (eye1[0], eye1[1]), (eye1[0]+eye1[2], eye1[1]+eye1[3]), (0, 255, 0), 2)
        cv2.rectangle(image, (eye2[0], eye2[1]), (eye2[0]+eye2[2], eye2[1]+eye2[3]), (0, 255, 0), 2)
    
    # Draw nose
    if len(nose) > 0:
        nose_position = nose[0]
        features['nose_position'] = (nose_position[0], nose_position[1])
        cv2.rectangle(image, (nose_position[0], nose_position[1]), (nose_position[0]+nose_position[2], nose_position[1]+nose_position[3]), (255, 0, 0), 2)
    
    # Draw mouth
    if len(mouth) > 0:
        mouth_position = mouth[0]
        features['mouth_position'] = (mouth_position[0], mouth_position[1])
        cv2.rectangle(image, (mouth_position[0], mouth_position[1]), (mouth_position[0]+mouth_position[2], mouth_position[1]+mouth_position[3]), (0, 0, 255), 2)
    
    # Calculate distances between the detected landmarks
    if 'eye_distance' in features and 'nose_position' in features:
        eye_to_nose_distance = np.linalg.norm(np.array([eye1[0], eye1[1]]) - np.array([nose_position[0], nose_position[1]]))
        features['eye_to_nose_distance'] = eye_to_nose_distance
    
    if 'nose_position' in features and 'mouth_position' in features:
        nose_to_mouth_distance = np.linalg.norm(np.array([nose_position[0], nose_position[1]]) - np.array([mouth_position[0], mouth_position[1]]))
        features['nose_to_mouth_distance'] = nose_to_mouth_distance
    
    return features, image

# Pick a random image from the dataset (in this case, male images)
random_image = random.choice(male_train_images)

# Extract geometric features and annotate the image
geometric_features, annotated_image = extract_geometric_features_and_display(random_image)

# Convert image to RGB for display with matplotlib
annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

# Display the annotated image
plt.imshow(annotated_image_rgb)
plt.axis('off')
plt.show()

# Print the geometric features
print("Geometric Features for the selected image:")
for key, value in geometric_features.items():
    print(f"{key}: {value}")


# In[39]:


# Define a function to extract geometric features and annotate them
def extract_geometric_features_and_display(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect eyes, nose, and mouth
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    nose = nose_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    mouth = mouth_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    features = {}
    
    # Draw eyes and calculate distance between them
    if len(eyes) >= 2:
        eye1, eye2 = eyes[0], eyes[1]
        eye_distance = np.linalg.norm(np.array([eye1[0], eye1[1]]) - np.array([eye2[0], eye2[1]]))
        features['eye_distance'] = eye_distance
        cv2.rectangle(image, (eye1[0], eye1[1]), (eye1[0]+eye1[2], eye1[1]+eye1[3]), (0, 255, 0), 2)
        cv2.rectangle(image, (eye2[0], eye2[1]), (eye2[0]+eye2[2], eye2[1]+eye2[3]), (0, 255, 0), 2)
    
    # Draw nose
    if len(nose) > 0:
        nose_position = nose[0]
        features['nose_position'] = (nose_position[0], nose_position[1])
        cv2.rectangle(image, (nose_position[0], nose_position[1]), (nose_position[0]+nose_position[2], nose_position[1]+nose_position[3]), (255, 0, 0), 2)
    
    # Draw mouth
    if len(mouth) > 0:
        mouth_position = mouth[0]
        features['mouth_position'] = (mouth_position[0], mouth_position[1])
        cv2.rectangle(image, (mouth_position[0], mouth_position[1]), (mouth_position[0]+mouth_position[2], mouth_position[1]+mouth_position[3]), (0, 0, 255), 2)
    
    # Calculate distances between the detected landmarks
    if 'eye_distance' in features and 'nose_position' in features:
        eye_to_nose_distance = np.linalg.norm(np.array([eye1[0], eye1[1]]) - np.array([nose_position[0], nose_position[1]]))
        features['eye_to_nose_distance'] = eye_to_nose_distance
    
    if 'nose_position' in features and 'mouth_position' in features:
        nose_to_mouth_distance = np.linalg.norm(np.array([nose_position[0], nose_position[1]]) - np.array([mouth_position[0], mouth_position[1]]))
        features['nose_to_mouth_distance'] = nose_to_mouth_distance
    
    return features, image

# Pick a random image from the dataset (in this case, male images)
random_image = random.choice(female_train_images)

# Extract geometric features and annotate the image
geometric_features, annotated_image = extract_geometric_features_and_display(random_image)

# Convert image to RGB for display with matplotlib
annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

# Display the annotated image
plt.imshow(annotated_image_rgb)
plt.axis('off')
plt.show()

# Print the geometric features
print("Geometric Features for the selected image:")
for key, value in geometric_features.items():
    print(f"{key}: {value}")


# In[1]:


import cv2
import os
import numpy as np

# Set paths for training data
train_path = r'C:\Users\Abhineswari\Downloads\archive (3)\Training'

# Get list of male and female directories
male_train_dir = os.path.join(train_path, 'male')
female_train_dir = os.path.join(train_path, 'female')

# Read image filenames
male_images = [os.path.join(male_train_dir, img) for img in os.listdir(male_train_dir) if img.endswith('.jpg')]
female_images = [os.path.join(female_train_dir, img) for img in os.listdir(female_train_dir) if img.endswith('.jpg')]

# Load Haar Cascade for face and feature detection
face_cascade = cv2.CascadeClassifier(r"C:\Users\Abhineswari\Downloads\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(r"C:\Users\Abhineswari\Downloads\haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier(r"C:\Users\Abhineswari\Downloads\haarcascade_mcs_mouth.xml")
nose_cascade = cv2.CascadeClassifier(r"C:\Users\Abhineswari\Downloads\haarcascade_mcs_nose.xml")

# Function to detect faces and extract geometric features
def extract_geometric_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect eyes, nose, and mouth
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    nose = nose_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    mouth = mouth_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    features = {}

    # Calculate eye distance if both eyes are detected
    if len(eyes) >= 2:
        eye1, eye2 = eyes[0], eyes[1]
        eye_distance = np.linalg.norm(np.array([eye1[0], eye1[1]]) - np.array([eye2[0], eye2[1]]))
        features['eye_distance'] = eye_distance

        # Draw rectangles around the eyes (optional)
        cv2.rectangle(image, (eye1[0], eye1[1]), (eye1[0] + eye1[2], eye1[1] + eye1[3]), (0, 255, 0), 2)
        cv2.rectangle(image, (eye2[0], eye2[1]), (eye2[0] + eye2[2], eye2[1] + eye2[3]), (0, 255, 0), 2)
    else:
        features['eye_distance'] = 0  # Default if eyes are not detected

    if len(nose) > 0:
        nose_position = nose[0]
        features['nose_position'] = (nose_position[0], nose_position[1])
    else:
        features['nose_position'] = (0, 0)  # Default if nose is not detected

    if len(mouth) > 0:
        mouth_position = mouth[0]
        features['mouth_position'] = (mouth_position[0], mouth_position[1])
    else:
        features['mouth_position'] = (0, 0)  # Default if mouth is not detected

    # Calculate distances between the detected landmarks
    if 'eye_distance' in features and features['eye_distance'] > 0 and 'nose_position' in features:
        eye_to_nose_distance = np.linalg.norm(np.array([eye1[0], eye1[1]]) - np.array([features['nose_position'][0], features['nose_position'][1]]))
        features['eye_to_nose_distance'] = eye_to_nose_distance
    else:
        features['eye_to_nose_distance'] = 0  # Default if unable to compute

    if 'nose_position' in features and 'mouth_position' in features:
        nose_to_mouth_distance = np.linalg.norm(np.array([features['nose_position'][0], features['nose_position'][1]]) - np.array([features['mouth_position'][0], features['mouth_position'][1]]))
        features['nose_to_mouth_distance'] = nose_to_mouth_distance
    else:
        features['nose_to_mouth_distance'] = 0  # Default if unable to compute

    return features

# Function to calculate average geometric features for a set of images
def calculate_average_features(image_paths):
    feature_sums = {
        'eye_distance': 0,
        'eye_to_nose_distance': 0,
        'nose_to_mouth_distance': 0
    }
    counts = 0

    for img_path in image_paths:
        image = cv2.imread(img_path)
        features = extract_geometric_features(image)

        if 'eye_distance' in features and features['eye_distance'] > 0:
            feature_sums['eye_distance'] += features['eye_distance']
            counts += 1

        if 'eye_to_nose_distance' in features:
            feature_sums['eye_to_nose_distance'] += features['eye_to_nose_distance']

        if 'nose_to_mouth_distance' in features:
            feature_sums['nose_to_mouth_distance'] += features['nose_to_mouth_distance']

    # Calculate averages
    if counts > 0:
        averages = {k: v / counts for k, v in feature_sums.items()}
        return averages
    return None

# Calculate average features for male and female images
average_features_male = calculate_average_features(male_images)
average_features_female = calculate_average_features(female_images)

# Print the results
print("Average Geometric Features for Male:")
print(average_features_male)

print("\nAverage Geometric Features for Female:")
print(average_features_female)


# In[40]:


# Define a function to apply Sobel edge detection
def apply_sobel(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Sobel filter
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)  # Sobel in x-direction
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)  # Sobel in y-direction
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)  # Combine both
    return sobel_edges


# In[41]:


# Define a function to apply Canny edge detection
def apply_canny(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)  # You can adjust the thresholds
    return edges


# In[42]:


# Select a random male face to visualize
random_face = random.choice(male_faces)

# Apply Canny and Sobel edge detection
canny_edges = apply_canny(random_face)
sobel_edges = apply_sobel(random_face)

# Display the original face, Canny edges, and Sobel edges
plt.figure(figsize=(12, 6))

# Original face
plt.subplot(1, 3, 1)
plt.title("Original Face")
plt.imshow(cv2.cvtColor(random_face, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Canny edges
plt.subplot(1, 3, 2)
plt.title("Canny Edges")
plt.imshow(canny_edges, cmap='gray')
plt.axis('off')

# Sobel edges
plt.subplot(1, 3, 3)
plt.title("Sobel Edges")
plt.imshow(sobel_edges, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()


# In[43]:


# Function to extract geometric features, including jawline width
def extract_geometric_features(face_image):
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    
    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    
    if len(eyes) >= 2:
        eye1, eye2 = eyes[0], eyes[1]
        eye_distance = np.linalg.norm(np.array([eye1[0], eye1[1]]) - np.array([eye2[0], eye2[1]]))
    else:
        eye_distance = 0  # Default value if eyes are not detected

    # Estimate jawline width using face image dimensions
    jawline_width = face_image.shape[1]  # The width of the cropped face image

    return eye_distance, jawline_width

# Function to classify gender based on extracted features
def classify_gender(face_image):
    # Extract geometric features
    eye_distance, jawline_width = extract_geometric_features(face_image)

    # Classification rules
    # You may need to adjust these threshold values based on your dataset
    if jawline_width > 70:  # Example threshold for jawline width
        return 'Male'
    else:
        return 'Female'

# Apply the classification on cropped faces
male_count = 0
female_count = 0

for face in male_faces:
    gender = classify_gender(face)
    if gender == 'Male':
        male_count += 1
    else:
        female_count += 1

for face in female_faces:
    gender = classify_gender(face)
    if gender == 'Male':
        male_count += 1
    else:
        female_count += 1

# Print results
print(f'Male faces classified: {male_count}, Female faces classified: {female_count}')


# # Validation

# In[4]:


# Set thresholds based on average features
def set_thresholds(average_features_male, average_features_female):
    thresholds = {}
    for feature in average_features_male.keys():
        thresholds[feature] = (average_features_male[feature] + average_features_female[feature]) / 2
    return thresholds

# Define the classification function
def classify_gender(features, thresholds):
    eye_distance = features['eye_distance']
    eye_to_nose_distance = features['eye_to_nose_distance']
    nose_to_mouth_distance = features['nose_to_mouth_distance']

    if (eye_distance < thresholds['eye_distance'] and
        eye_to_nose_distance < thresholds['eye_to_nose_distance']):
        return 'Male'
    elif (eye_distance >= thresholds['eye_distance'] and
          eye_to_nose_distance >= thresholds['eye_to_nose_distance']):
        return 'Female'
    else:
        return 'Uncertain'

# Calculate thresholds
thresholds = set_thresholds(average_features_male, average_features_female)

# Test with a sample feature set
sample_features = {
    'eye_distance': 67,
    'eye_to_nose_distance': 36,
    'nose_to_mouth_distance': 35
}

# Classify based on the sample features
gender = classify_gender(sample_features, thresholds)
print(f"The classified gender for the sample features is: {gender}")


# In[7]:


def classify_and_annotate_images(validation_images, thresholds):
    for img_path in validation_images:
        # Read the image
        image = cv2.imread(img_path)
        features = extract_geometric_features(image)

        if features is not None:
            # Classify gender based on features
            gender = classify_gender(features, thresholds)

            # Annotate the image with the classification result
            label_text = f"Gender: {gender}"

            # Adjust text font, size, and position
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7  # Smaller font scale
            font_thickness = 2
            text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
            text_x = 20  # Position at the top-left corner
            text_y = image.shape[0] - 20  # Place closer to the bottom

            # Ensure the text is clearly visible with a background rectangle
            cv2.rectangle(image, (text_x - 5, text_y - text_size[1] - 5), 
                          (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), cv2.FILLED)
            cv2.putText(image, label_text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)

            # Convert to RGB for displaying with matplotlib
            annotated_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Display the image with the label using matplotlib
            plt.imshow(annotated_image_rgb)
            plt.title(f"Classified as: {gender}")
            plt.axis('off')
            plt.show()

# Run the function to classify and display validation images with proper annotations
classify_and_annotate_images(validation_images, thresholds)


# In[ ]:





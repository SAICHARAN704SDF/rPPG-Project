import cv2
import torch
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt, welch
from modelrythm import RhythmMamba 

# 1. Filter: Standard rPPG range (0.7Hz to 3.0Hz = 42 to 180 BPM)
def butter_bandpass(data, fs=30):
    nyq = 0.5 * fs
    b, a = butter(4, [0.7/nyq, 3.0/nyq], btype='band')
    return filtfilt(b, a, data)

# 2. Load Model on Mac CPU
DEVICE = torch.device("cpu")
model = RhythmMamba()
try:
    # Use weights_only=True for safety
    model.load_state_dict(torch.load("rythmmamba.pth", map_location=DEVICE, weights_only=True))
    model.eval()
    print(" Model loaded successfully on Mac.")
except Exception as e:
    print(f" Error loading model: {e}")
    exit()

# 3. Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0) # Open Mac Webcam
frame_buffer = []

print("System Ready. Please stay still and look at the camera...")

while cap.isOpened():
    success, frame = cap.read()
    if not success: 
        print("Failed to grab frame from camera.")
        break
    
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        # Access the first face in the list
        face_landmarks = results.multi_face_landmarks[0]
        
        # Landmark 10 is the center of the forehead
        lm = face_landmarks.landmark[10]
        cx, cy = int(lm.x * w), int(lm.y * h)
        
        # Crop 64x64 ROI (Region of Interest)
        y1, y2, x1, x2 = cy-32, cy+32, cx-32, cx+32
        
        if y1 > 0 and x1 > 0 and y2 < h and x2 < w:
            roi = rgb_frame[y1:y2, x1:x2]
            roi = cv2.resize(roi, (64, 64))
            frame_buffer.append(roi)
            
            # Maintain sliding window of 128 frames (approx 4 seconds at 30fps)
            if len(frame_buffer) > 128:
                frame_buffer.pop(0)

            if len(frame_buffer) == 128:
                # Preprocess: Temporal Difference (Matches Training)
                seq = np.array(frame_buffer).astype(np.float32)
                diff = (seq[1:] - seq[:-1]) / (seq[1:] + seq[:-1] + 1e-6)
                diff = np.pad(diff, ((1, 0), (0, 0), (0, 0), (0, 0)), mode='constant')
                diff = (diff - np.mean(diff)) / (np.std(diff) + 1e-6)
                
                # Permute to [Batch=1, Time=128, Channels=3, Height=64, Width=64]
                input_tensor = torch.from_numpy(diff).unsqueeze(0).permute(0, 1, 4, 2, 3)
                
                with torch.no_grad():
                    # Predict waveform
                    pred_wave = model(input_tensor).numpy().flatten()
                    
                    # 4. Signal Processing: Filter -> Welch/FFT -> BPM
                    clean_wave = butter_bandpass(pred_wave)
                    freqs, psd = welch(clean_wave, fs=30, nperseg=128)
                    
                    # Focus on human heart rate frequencies
                    valid = (freqs >= 0.7) & (freqs <= 3.0)
                    if np.any(valid):
                        bpm = freqs[valid][np.argmax(psd[valid])] * 60
                        # Visual output on the camera frame
                        cv2.putText(frame, f"BPM: {int(bpm)}", (cx-50, cy-50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # Draw a box on the forehead for visual confirmation
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Face Detected", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the webcam window
    cv2.imshow('Live Heart Rate Monitor', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


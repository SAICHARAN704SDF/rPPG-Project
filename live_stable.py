import cv2
import torch
import numpy as np
import mediapipe as mp
from collections import deque

from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import welch

from modelrythm import RhythmMamba

FPS = 29.26

BUFFER_SIZE = 256

IMG_SIZE = 72

DEVICE = torch.device("cpu")


def butter_bandpass(data, fs=FPS):

    low = 0.7
    high = 3.0

    nyq = 0.5 * fs

    b, a = butter(
        4,
        [low / nyq, high / nyq],
        btype='band'
    );

    return filtfilt(b, a, data)


def estimate_bpm(signal):

    signal = signal - np.mean(signal)

    signal = butter_bandpass(signal)

    freqs, psd = welch(
        signal,
        fs=FPS,
        nperseg=min(256, len(signal))
    )

    valid = (
        (freqs >= 0.7)
        &
        (freqs <= 3.0)
    )

    freqs = freqs[valid]
    psd = psd[valid]

    peak_idx = np.argmax(psd)

    peak_freq = freqs[peak_idx]

    bpm = peak_freq * 60

    confidence = psd[peak_idx]

    return bpm, confidence, signal

model = RhythmMamba()

model.load_state_dict(
    torch.load(
        "rhythmmamba_best.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

print("Model loaded")


mp_face = mp.solutions.face_detection

face_detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)


cap = cv2.VideoCapture(0)

frame_buffer = deque(maxlen=BUFFER_SIZE)

bpm_history = deque(maxlen=10)

display_bpm = 0

print("Starting live BPM monitor...")


while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_detector.process(rgb)

    if results.detections:

        detection = results.detections[0]

        bbox = (
            detection
            .location_data
            .relative_bounding_box
        )

        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)

        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        x2 = x1 + bw
        y2 = y1 + bh

        # clamp
        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(w, x2)
        y2 = min(h, y2)

        face = rgb[y1:y2, x1:x2]

        if face.size > 0:

            # resize
            face = cv2.resize(
                face,
                (IMG_SIZE, IMG_SIZE)
            )

            face = face.astype(
                np.float32
            ) / 255.0

            frame_buffer.append(face)

            # draw ROI
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0,255,0),
                2
            )


    if len(frame_buffer) == BUFFER_SIZE:

        seq = np.array(frame_buffer)

        # [T,H,W,C]
        seq = np.transpose(
            seq,
            (0,3,1,2)
        )

        # [1,T,C,H,W]
        x = torch.tensor(
            seq,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            pred = model(x)

        pred = (
            pred
            .squeeze()
            .cpu()
            .numpy()
        )

        try:

            bpm, confidence, clean_wave = estimate_bpm(pred)

            if confidence > 0.001:

                bpm_history.append(bpm)

               
                display_bpm = int(
                    np.median(bpm_history)
                )

        except:
            pass


        wave = clean_wave[-200:]

        wave = (
            wave - np.min(wave)
        ) / (
            np.max(wave)
            - np.min(wave)
            + 1e-6
        )

        wave = (wave * 100).astype(np.int32)

        for i in range(1, len(wave)):

            cv2.line(
                frame,
                (i*3, 400 - wave[i-1]),
                (i*3, 400 - wave[i]),
                (0,255,255),
                2
            )

   
    cv2.putText(
        frame,
        f"BPM: {display_bpm}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0,255,0),
        3
    )

    cv2.putText(
        frame,
        "Stay Still",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    cv2.imshow(
        "Live rPPG Monitor",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
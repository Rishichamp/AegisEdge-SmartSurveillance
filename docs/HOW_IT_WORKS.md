# Algorithmic Deep Dive — How It Works
=======================================

This document provides a novice-friendly explanation of the computer vision algorithms, mathematical equations, and spatial-temporal reasoning logic used in AegisEdge.

---

## 1. Real-Time Tracking (YOLOv8 + ByteTrack)

### 1.1. Bounding Box Regression
When the system receives a frame, it passes it to **YOLOv8** (You Only Look Once). YOLOv8 splits the image into a grid and predicts bounding boxes as coordinates:
$$\text{Bbox} = [x_1, y_1, x_2, y_2]$$
where $(x_1, y_1)$ is the top-left corner, and $(x_2, y_2)$ is the bottom-right corner.

### 1.2. Interframe Tracking (ByteTrack)
Detectors process frames independently. They do not know that "Person A" in Frame 1 is the same "Person A" in Frame 2. 

**ByteTrack** solves this by:
1. Storing active tracks with Kalman Filters (which predict where a box *should* move based on past speed).
2. Calculating the **Intersection over Union (IoU)** overlap between the predicted boxes and new detections:
   $$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}$$
3. Matching detections using the Hungarian matching algorithm. A high IoU associates the box with its persistent `track_id`.

---

## 2. Fight Detection (Optical Flow + Geometry)

Altercations are characterized by two elements: close proximity of actors and fast, chaotic pixel motion.

### 2.1. Farneback Dense Optical Flow
To measure pixel motion, we use **Farneback's Dense Optical Flow**. It calculates a displacement vector $(dx, dy)$ for *every single pixel* between two consecutive grayscale images:

```
Frame t-1   ───┐
               ├──> [Farneback Algorithm] ──> Flow Field (dx, dy) for each pixel
Frame t     ───┘
```

For each pixel, we compute the magnitude of movement:
$$\text{Magnitude}(x,y) = \sqrt{dx(x,y)^2 + dy(x,y)^2}$$

We average the magnitudes across the entire frame. If the average exceeds our config threshold (e.g. `30.0` pixels change), it flags high-energy activity.

### 2.2. Proximity Scoring
If 2+ people are detected, we calculate the pairwise Euclidean distances between the centers of their bounding boxes. For center coordinates $(x_a, y_a)$ and $(x_b, y_b)$:
$$\text{Distance} = \sqrt{(x_a - x_b)^2 + (y_a - y_b)^2}$$

If the distance is less than our proximity threshold (e.g., `150` pixels), the rule engine flags the pair as in contact.

---

## 3. Restricted Zones (Ray-Casting Algorithm)

Security cameras monitor complex shapes (e.g., corridors, gates). AegisEdge supports custom polygonal zones.

### 3.1. ray-Casting Point-in-Polygon (PIP)
To check if a person is in a zone, we extract their feet coordinates (the bottom center of their bounding box):
$$\text{Feet} = \left( \frac{x_1 + x_2}{2}, \ y_2 \right)$$

We then run the **Ray-Casting Algorithm**:
1. Cast an imaginary horizontal ray starting from the feet point extending infinitely to the right.
2. Count how many times this ray intersects the boundary edges of the zone polygon.
3. **The Rule:** If the number of intersections is **odd**, the point is inside the polygon. If the number of intersections is **even**, the point is outside.

```
       Polygon Zone
     ┌──────────────┐
     │  ● (Inside)  │ ───(ray)───> [1 Intersection = Odd -> INSIDE]
     │              │
  ●  └──────────────┘ ───(ray)───> [2 Intersections = Even -> OUTSIDE]
(Outside)
```

---

## 4. Theft Detection (Spatial-Temporal State Machine)

Shoplifting or office theft is represented as an object disappearing while a person is nearby.

### 4.1. The Disappearance State Machine
1. **Track Assets:** The system monitors stationary categories (laptops, bags) and saves their locations.
2. **Associate Owners:** It calculates the distance between the asset and nearby people. The closest person within the threshold (e.g., `120` pixels) is mapped as the "associated entity".
3. **Detect Disappearance:** If the asset is missing in the detection list:
   - It increments a `frames_missing` counter.
   - If `frames_missing` reaches `15` frames (0.5 seconds), it confirms the item was removed rather than momentarily blocked by someone walking in front of it.
   - If confirmed, it triggers a critical alert: *Asset stolen by associated entity ID.*

---

## 5. Sequence Posture Recognition (MobileNetV3 + LSTM)

To recognize complex actions (like crouching, picking, punching), we use a deep-learning sequence model.

### 5.1. Feature Extraction (MobileNetV3)
For each frame, the pipeline crops the person's bounding box and resizes it to $224 \times 224$. It passes the crop through the **MobileNetV3** backbone. 
Instead of predicting a class, we extract the activations from the global average pooling layer. This yields a **576-dimensional feature vector** representing the visual posture of the person.

### 5.2. Recurrent sequence modeling (LSTM)
We feed these 576-dimensional vectors into a sliding queue buffer of length 16 (representing the last 0.5 seconds of the actor's history).
This $16 \times 576$ tensor is fed to our **LSTM (Long Short-Term Memory)** network. 

The LSTM processes the frames step-by-step, maintaining an internal hidden memory vector $h_t$ that gets updated at each step. The final hidden state $h_{16}$ is passed to fully connected layers to output probabilities for our target classes (normal, fight, theft, intrusion).

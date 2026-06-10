# Hand AR Engine

Control a virtual 3D object with nothing but your hand.

HandSpace AR is an experimental augmented reality interaction system that explores multiple ways humans can manipulate digital objects in physical space. Using computer vision, hand tracking, and custom 3D geometry, the project transforms a webcam into a gesture-driven AR environment where virtual objects can be rotated, grabbed, moved, and manipulated naturally.

No controllers.
No markers.
Just your hand.

---

## 🎥 Demo Concept

Imagine a floating cube sitting in front of you.

You can:

🖐️ Grab it with your fingers

🤏 Rotate it like a physical object

✋ Move it around the screen

📡 Control it remotely from a distance

✂️ Instantly disconnect using a scissor gesture

The system continuously tracks your hand and intelligently switches between interaction modes depending on how you approach the object.

---

# 🚀 Features

## 📡 Wireless Control Mode

Control the cube without touching it.

The system estimates:

- Roll
- Pitch
- Twist (Yaw)

directly from your hand orientation and maps them onto the virtual object.

Move your hand.

The cube follows.

No contact required.

---

## 🤏 Tactile Rotation Mode

When two fingers touch the object, the interaction changes completely.

The cube becomes physically "grabbable."

Features:

- Two-point rotational control
- Dynamic object alignment
- Real-time orientation reconstruction
- Natural object manipulation
- Physical grip simulation

Instead of controlling Euler angles directly, the cube aligns itself to the actual 3D geometry formed by your fingers.

---

## ✋ Drag Mode

Close your hand near the cube and drag it through space.

The object follows your hand with smooth motion filtering for a more natural feel.

Perfect for:

- Repositioning
- Object transport
- AR workspace interaction

---

## ✂️ Gesture-Based Disconnect

Wireless mode can be instantly terminated using a scissor gesture.

This creates a simple, intuitive way to release control without touching menus or pressing keys.

---

## 👁️ Hand Occlusion

One of the biggest immersion killers in simple AR systems is incorrect rendering order.

This project solves that.

Your hand correctly appears in front of the cube whenever it should.

The result is a much more convincing illusion that the virtual object exists inside the real world.

---

# 🧠 Computer Vision Pipeline

The system uses:

### Hand Tracking
- 21-point hand skeleton tracking
- Finger state detection
- Gesture recognition
- Orientation estimation

### Interaction Analysis
- Contact detection
- Grip detection
- Finger pair selection
- Mode switching state machine

### Motion Processing
- Exponential smoothing
- Dynamic confidence weighting
- Noise filtering
- Stability estimation

The result is a surprisingly stable interaction system despite using only a single RGB webcam.

---

# 🧊 Custom 3D Engine

The cube renderer is built from scratch.

Features include:

- 3D vertex transformations
- Rotation matrices
- Euler rotations
- Basis-vector alignment
- Perspective-style projection
- Face highlighting
- Dynamic transparency
- Real-time rendering

No game engine required.

---

# 🎮 Interaction Modes

| Mode | Description |
|--------|--------|
| Disconnected | Cube remains idle |
| Wireless | Hand controls cube orientation remotely |
| Dragging | Hand physically moves cube |
| Tactile Rotation | Two-finger grip rotates cube |
| Gesture Release | Instantly disconnects control |

---

# 🛠️ Tech Stack

### Computer Vision
- OpenCV
- MediaPipe

### Math & Geometry
- NumPy
- Linear Algebra
- Rotation Matrices
- Vector Mathematics

### Concepts Used
- Augmented Reality
- Human-Computer Interaction
- Gesture Recognition
- State Machines
- Spatial Tracking
- Motion Filtering
- 3D Transformations

---

# 🎯 Why I Built This

Most webcam hand-tracking demos stop at recognizing gestures.

I wanted to go further.

The goal was to explore how virtual objects could feel genuinely manipulable using only natural hand movement.

This project became an experiment in:

- AR interaction design
- Spatial computing
- Touchless interfaces
- Digital object manipulation
- Human-computer interaction

while building every major component from scratch.

---

# 🔮 Future Ideas

- Multiple object support
- Physics interactions
- Virtual buttons and UI
- Full 6DOF object manipulation
- Multi-hand collaboration
- Hand force estimation
- AR assembly tasks
- VR headset integration
- Neural gesture recognition
- Custom object importing

---

> "The future of computing isn't clicking buttons.
>
> It's reaching out and touching information directly."

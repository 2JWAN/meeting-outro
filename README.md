# Work Meeting Outro

Because clicking "Leave meeting" is way too normal.

A gesture-controlled webcam effect that lets you disappear and reappear during online meetings.

Built with Python, OpenCV, MediaPipe, and OBS.

## Features

- Peace sign: reaction video + disappear/reappear
- Finger snap: disappear/reappear
- Smooth transition effect
- Works with Google Meet through OBS Virtual Camera

## Setup

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. Move out of the camera frame.
2. Press `B` to capture the empty background.
3. Return to the frame.
4. Use a peace sign or finger snap to disappear/reappear.
5. Press `Q` to quit.

That's it. Have a dramatic meeting exit.

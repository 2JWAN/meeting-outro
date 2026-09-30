import cv2
import mediapipe as mp
import time
import numpy as np


# =========================================================
# SETTINGS
# =========================================================

TRANSITION_DURATION = 0.75
PEACE_HOLD_TIME = 0.30

REACTION_FILE = "reaction.mp4"
REACTION_WIDTH_RATIO = 0.42


# -------------------------
# SNAP SETTINGS
# -------------------------

# How close thumb + middle finger must be
# before we consider the snap "armed".
SNAP_TOUCH_DISTANCE = 0.055

# How far apart they must become afterward.
SNAP_RELEASE_DISTANCE = 0.11

# Entire snap must happen within this time.
SNAP_MAX_TIME = 0.40

# Prevent one snap from firing several times.
SNAP_COOLDOWN = 1.0

# Minimum downward middle-finger movement.
SNAP_MIN_MOTION = 0.035


# =========================================================
# MEDIAPIPE
# =========================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")


# =========================================================
# REACTION VIDEO
# =========================================================

reaction = cv2.VideoCapture(REACTION_FILE)

if not reaction.isOpened():
    raise RuntimeError(
        f"Could not open {REACTION_FILE}"
    )

reaction_fps = reaction.get(
    cv2.CAP_PROP_FPS
)

if reaction_fps <= 0:
    reaction_fps = 30

reaction_frame_count = int(
    reaction.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

print(
    f"Reaction loaded: "
    f"{reaction_frame_count} frames @ "
    f"{reaction_fps:.1f} FPS"
)


# =========================================================
# GESTURE FUNCTIONS
# =========================================================

def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y


def is_peace_sign(hand_landmarks):

    lm = hand_landmarks.landmark

    index_up = finger_up(
        lm, 8, 6
    )

    middle_up = finger_up(
        lm, 12, 10
    )

    ring_up = finger_up(
        lm, 16, 14
    )

    pinky_up = finger_up(
        lm, 20, 18
    )

    return (
        index_up
        and middle_up
        and not ring_up
        and not pinky_up
    )

# snap
def distance(a, b):
    """
    Distance between two MediaPipe
    landmarks.
    """

    return np.sqrt(
        (a.x - b.x) ** 2
        +
        (a.y - b.y) ** 2
    )


# =========================================================
# REACTION OVERLAY
# =========================================================

def overlay_reaction(
    background_frame,
    reaction_frame,
    opacity=1.0
):

    output = background_frame.copy()

    camera_h, camera_w = (
        output.shape[:2]
    )

    rh, rw = (
        reaction_frame.shape[:2]
    )

    target_w = int(
        camera_w
        * REACTION_WIDTH_RATIO
    )

    scale = (
        target_w / rw
    )

    target_h = int(
        rh * scale
    )

    reaction_resized = cv2.resize(
        reaction_frame,
        (
            target_w,
            target_h
        )
    )

    # Center horizontally
    x = (
        camera_w
        - target_w
    ) // 2

    # Slightly toward top
    y = int(
        camera_h * 0.08
    )


    # Make sure reaction fits
    if (
        y + target_h
        > camera_h
    ):

        target_h = (
            camera_h - y
        )

        reaction_resized = cv2.resize(
            reaction_resized,
            (
                target_w,
                target_h
            )
        )


    roi = output[
        y:y + target_h,
        x:x + target_w
    ]


    blended = cv2.addWeighted(
        reaction_resized,
        opacity,
        roi,
        1.0 - opacity,
        0
    )


    output[
        y:y + target_h,
        x:x + target_w
    ] = blended


    return output


# =========================================================
# MAIN STATE
# =========================================================

visible = True

background = None


# -------------------------
# PEACE STATE
# -------------------------

peace_started = None
peace_locked = False


# -------------------------
# TRANSITION STATE
# -------------------------

transitioning = False

transition_start = 0

transition_from_visible = True

transition_frame = None


# -------------------------
# REACTION STATE
# -------------------------

play_reaction = False


# -------------------------
# SNAP STATE
# -------------------------

snap_armed = False

snap_arm_time = 0.0

snap_middle_start_y = 0.0

last_snap_time = 0.0


# =========================================================
# STARTUP MESSAGE
# =========================================================

print()
print("================================")
print("        VANISH CAMERA")
print("================================")
print()
print("B   = Capture empty background")
print("Q   = Quit")
print()
print("================================")
print()


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    success, camera_frame = (
        cap.read()
    )

    if not success:
        break


    # Mirror like FaceTime / Meet
    camera_frame = cv2.flip(
        camera_frame,
        1
    )


    # =====================================================
    # HAND TRACKING
    # =====================================================

    rgb = cv2.cvtColor(
        camera_frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    peace_detected = False
    snap_triggered = False

    now = time.time()


    if results.multi_hand_landmarks:

        for hand_landmarks in (
            results.multi_hand_landmarks
        ):

            lm = (
                hand_landmarks.landmark
            )


            # =============================================
            # PEACE DETECTION
            # =============================================

            if is_peace_sign(
                hand_landmarks
            ):

                peace_detected = True


            # =============================================
            # SNAP DETECTION
            # =============================================

            thumb_tip = lm[4]
            middle_tip = lm[12]


            touch_distance = distance(
                thumb_tip,
                middle_tip
            )


            # ---------------------------------------------
            # STAGE 1:
            #
            # Thumb + middle finger touch.
            # ---------------------------------------------

            if (
                not snap_armed
                and
                touch_distance
                < SNAP_TOUCH_DISTANCE
                and
                now - last_snap_time
                > SNAP_COOLDOWN
            ):

                snap_armed = True

                snap_arm_time = now

                snap_middle_start_y = (
                    middle_tip.y
                )



            # ---------------------------------------------
            # STAGE 2:
            #
            # Middle finger releases and
            # rapidly moves downward.
            # ---------------------------------------------

            elif snap_armed:

                elapsed_snap = (
                    now
                    - snap_arm_time
                )


                middle_motion = (
                    middle_tip.y
                    - snap_middle_start_y
                )


                released = (
                    touch_distance
                    > SNAP_RELEASE_DISTANCE
                )


                moved_fast = (
                    middle_motion
                    > SNAP_MIN_MOTION
                )


                if (
                    released
                    and moved_fast
                    and
                    elapsed_snap
                    <= SNAP_MAX_TIME
                ):

                    snap_triggered = True

                    snap_armed = False

                    last_snap_time = now

                   


                # Took too long:
                # cancel the snap.
                elif (
                    elapsed_snap
                    > SNAP_MAX_TIME
                ):

                    snap_armed = False


    # =====================================================
    # PEACE TRIGGER
    # =====================================================

    if peace_detected:

        if peace_started is None:

            peace_started = (
                time.time()
            )


        held_time = (
            time.time()
            - peace_started
        )


        if (
            held_time
            >= PEACE_HOLD_TIME
            and
            not peace_locked
            and
            not transitioning
        ):

            peace_locked = True


            if background is None:

                print(
                    "⚠ Capture background "
                    "first with B"
                )


            else:

                transitioning = True

                transition_start = (
                    time.time()
                )

                transition_from_visible = (
                    visible
                )

                transition_frame = (
                    camera_frame.copy()
                )


                # =========================================
                # Play meme only at piece out start not back
                # =========================================

                if visible:

                    play_reaction = True

                    reaction.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        0
                    )

                   


                else:

                    play_reaction = False

                    


    else:

        peace_started = None
        peace_locked = False


    # =====================================================
    # SNAP TRIGGER
    # =====================================================

    if (
        snap_triggered
        and
        not transitioning
    ):

        if background is None:

            print(
                "Capture background "
                "first with B"
            )


        else:

            transitioning = True

            transition_start = (
                time.time()
            )

            transition_from_visible = (
                visible
            )

            transition_frame = (
                camera_frame.copy()
            )


            # =============================================
            # IMPORTANT:
            #
            # Snap NEVER plays the meme.
            # =============================================

            play_reaction = False


           

    # =====================================================
    # TRANSITION
    # =====================================================

    if transitioning:

        elapsed = (
            time.time()
            - transition_start
        )


        progress = min(
            elapsed
            / TRANSITION_DURATION,
            1.0
        )


        # Smoothstep:
        #
        # smoother than a simple
        # linear fade.

        smooth = (
            progress
            * progress
            * (
                3.0
                - 2.0 * progress
            )
        )


        # =================================================
        # VANISH
        # =================================================

        if transition_from_visible:

            output = cv2.addWeighted(
                transition_frame,
                1.0 - smooth,
                background,
                smooth,
                0
            )


        # =================================================
        # REAPPEAR
        # =================================================

        else:

            output = cv2.addWeighted(
                camera_frame,
                smooth,
                background,
                1.0 - smooth,
                0
            )


        # =================================================
        # PEACE  MEME REACTION VIDEO
        # =================================================

        if play_reaction:

            ok, reaction_frame = (
                reaction.read()
            )


            if ok:

                # -----------------------------------------
                # Reaction fade in
                # -----------------------------------------

                if progress < 0.20:

                    reaction_opacity = (
                        progress
                        / 0.20
                    )


                # -----------------------------------------
                # Reaction fade out
                # -----------------------------------------

                elif progress > 0.75:

                    reaction_opacity = (
                        1.0
                        -
                        (
                            progress
                            - 0.75
                        )
                        / 0.25
                    )


                else:

                    reaction_opacity = 1.0


                reaction_opacity = max(
                    0.0,
                    min(
                        reaction_opacity,
                        1.0
                    )
                )


                output = overlay_reaction(
                    output,
                    reaction_frame,
                    reaction_opacity
                )


        # =================================================
        # TRANSITION COMPLETE
        # =================================================

        if progress >= 1.0:

            transitioning = False

            visible = (
                not transition_from_visible
            )

            play_reaction = False


    # =====================================================
    # NORMAL OUTPUT
    # =====================================================

    else:

        if visible:

            output = (
                camera_frame.copy()
            )


        elif background is not None:

            output = (
                background.copy()
            )


        else:

            output = (
                camera_frame.copy()
            )


    # =====================================================
    # DISPLAY
    # =====================================================

    cv2.imshow(
        "Vanish Camera",
        output
    )


    key = (
        cv2.waitKey(1)
        & 0xFF
    )


    # =====================================================
    # KEYBOARD (Q for Quit / B for BG capture)
    # =====================================================

    if key == ord("q"):

        break


    elif key == ord("b"):

        background = (
            camera_frame.copy()
        )



# =========================================================
# CLEANUP
# =========================================================

reaction.release()

cap.release()

hands.close()

cv2.destroyAllWindows()
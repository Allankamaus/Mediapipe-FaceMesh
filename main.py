import cv2
import mediapipe as mp

#initialize Mediapip face mesh
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(max_num_hands=2)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)


#webcam
cap = cv2.VideoCapture(0)

use_face = True
use_hands = False
print("Press 'f' for Face Mesh, 'h' for Hand Tracking, 'q' to quit")



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignorning empty cam")
        break

    #convert bgr to rgb
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #disable writable flag for performance
    rgb_frame.flags.writeable = False


    if use_face:
        results = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for lm in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, lm,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

    elif use_hands:
        results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            height, width, _ = frame.shape
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                def fingers_up(hand_landmarks, image_height,image_width):
                    fingers = []
     #get landmark for each finger
                    lm = hand_landmarks.landmark

                    def y(id):
                        return int(lm[id].y * image_height)
    
                    def x(id):
                        return int(lm[id].x *image_width)
    
    #index finger
                    if y(8) < y(6) and y(7) < y(6):
                        fingers.append("index")

    #middle finger
                    if y(12) < y(10) :
                        fingers.append("middle")
    
    #ring finger
                    if y(16) < y(14):
                        fingers.append("ring")

    #pinky
                    if y(20) < y(18):
                        fingers.append("pinky")

    #thumb
                    if x(4) < x(3):
                        fingers.append("thumb")

                    return fingers

                fingers = fingers_up(hand_landmarks, height, width)
    
                if len(fingers) == 1:
                    print("One finger up")

                elif len(fingers) == 2:
                    print("two finger up")

                elif len(fingers) == 3:
                    print("3 finger up")

                elif len(fingers) == 4:
                    print("4 fingers up")

                elif len(fingers) == 5:
                    print("5 fingered")

                else:print("no fingering")

                if fingers == ["index"]:
                    print("index finger is up")

                elif fingers == ["middle"]:
                    print("middle finger is up")

    
    mode_text = "Mode: Face Mesh" if use_face else "Mode: Hand Tracking"
    cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Toggle Tracker", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('f'):
        use_face = True
        use_hands = False
    elif key == ord('h'):
        use_face = False
        use_hands = True
    elif key == ord('q') or key == 27:
        break
    elif key == ord('g'):
        use_face = True
        use_hands = True
        
cap.release()
cv2.destroyAllWindows()



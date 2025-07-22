import cv2
import mediapipe as mp

#initialize Mediapip face mesh
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


#initialize face mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode = False,
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5

)

#initialize hands mesh
hands = mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 2,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)

#webcam
cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignorning empty cam")
        break

    #convert bgr to rgb
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #disable writable flag for performance
    rgb_frame.flags.writeable = False
    results = face_mesh.process(rgb_frame)
    results2 = hands.process(rgb_frame)

    #set frame back to writebale drawing
    rgb_frame.flags.writeable = True
    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            #draw the face mesh on the frame
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections = mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style()
            )

    if results2.multi_hand_landmarks:
        for hand_landmarks in results2.multi_hand_landmarks:
            #draw hand mesh
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=hand_landmarks,
                connections = mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
            )

    cv2.imshow('Face Mesh',frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
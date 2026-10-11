import cv2
import mediapipe as mp
import time


class FaceMeshDetector:
    def __init__(self, staticMode=False, maxFaces=2,
                 minDetectionCon=0.5, minTrackingCon=0.5):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackingCon = minTrackingCon

        self.mpFaceMesh = mp.solutions.face_mesh      
        self.mpDraw = mp.solutions.drawing_utils

        self.facemesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,             
            max_num_faces=self.maxFaces,                    
            min_detection_confidence=self.minDetectionCon, 
            min_tracking_confidence=self.minTrackingCon    
        )

        self.drawspec = self.mpDraw.DrawingSpec(            
            thickness=1,
            circle_radius=2
        )

    def findFaceMesh(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.facemesh.process(imgRGB)        

        faces = []

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(             
                        img,
                        faceLms,
                        self.mpFaceMesh.FACEMESH_CONTOURS,
                        self.drawspec,
                        self.drawspec
                    )

                face = []

                for id, lm in enumerate(faceLms.landmark):

                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)

                    face.append([x, y])

                faces.append(face)                          

        return img, faces



def main():

    cap = cv2.VideoCapture(0)

    pTime = 0

    detector = FaceMeshDetector(maxFaces=2)               

    while True:

        success, img = cap.read()

        if not success:
            break

        img, faces = detector.findFaceMesh(img)

        if len(faces) != 0:
            print(len(faces[0]))  # Prints 468 landmarks

        cTime = time.time()

        fps = 1 / (cTime - pTime)                           
        pTime = cTime                                       

        cv2.putText(
            img,
            f"FPS: {int(fps)}",
            (20, 70),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (0, 255, 0),
            3,
        )

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):               
            break

    cap.release()
    cv2.destroyAllWindows()


# <-- YOUR MISTAKE (Never assign __name__)
if __name__ == "__main__":
    main()

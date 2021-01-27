import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float,img_as_ubyte
import dlib

detector = dlib.get_frontal_face_detector()
predector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def add_photo(img,pt2,mask):
    mask=img_as_float(mask)
    img=img_as_float(img)
    pt1=np.float32([[0,0],
                  [mask.shape[1],0],
                  [0,mask.shape[0]],
                  [mask.shape[1],mask.shape[0]]
                  ])
    mat = cv2.getPerspectiveTransform(pt1,pt2)
    res=cv2.warpPerspective(mask,mat,(img.shape[1],img.shape[0]),cv2.INTER_LINEAR,cv2.BORDER_CONSTANT,borderValue=(-1, -1, -1))

    return res

def pro(img,mask,draw_rect1=True,draw_rect2=True,draw_lines=True,draw_mask=True):
    copy = img.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces=detector(gray, 0)

    for face in faces:
        x1 = face.left()
        y1 =face.top()
        x2= face.right()
        y2= face.bottom()
        landmarks = predector(gray,face)

        size = copy.shape
        #2D image points. If you change the image, you need to change vector
        image_points = np.array([
                                (landmarks.part(33).x,landmarks.part(33).y),     # Nose tip
                                (landmarks.part(8).x,landmarks.part(8).y),       # Chin
                                (landmarks.part(36).x,landmarks.part(36).y),     # Left eye left corner
                                (landmarks.part(45).x,landmarks.part(45).y),     # Right eye right corne
                                (landmarks.part(48).x,landmarks.part(48).y),     # Left Mouth corner
                                (landmarks.part(54).x,landmarks.part(54).y)      # Right mouth corner
                            ], dtype="double")

        # 3D model points.
        model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner

                            ])
        # Camera internals
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )

        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

        (b1, jacobian) = cv2.projectPoints(np.array([(350.0, 270.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        (b2, jacobian) = cv2.projectPoints(np.array([(-350.0, -270.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        (b3, jacobian) = cv2.projectPoints(np.array([(-350.0, 270, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        (b4, jacobian) = cv2.projectPoints(np.array([(350.0, -270.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        (b11, jacobian) = cv2.projectPoints(np.array([(450.0, 350.0, 400.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        (b12, jacobian) = cv2.projectPoints(np.array([(-450.0, -350.0, 400.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        (b13, jacobian) = cv2.projectPoints(np.array([(-450.0, 350, 400.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        (b14, jacobian) = cv2.projectPoints(np.array([(450.0, -350.0, 400.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        b1 = ( int(b1[0][0][0]), int(b1[0][0][1]))
        b2 = ( int(b2[0][0][0]), int(b2[0][0][1]))
        b3 = ( int(b3[0][0][0]), int(b3[0][0][1]))
        b4 = ( int(b4[0][0][0]), int(b4[0][0][1]))

        b11 = ( int(b11[0][0][0]), int(b11[0][0][1]))
        b12 = ( int(b12[0][0][0]), int(b12[0][0][1]))
        b13 = ( int(b13[0][0][0]), int(b13[0][0][1]))
        b14 = ( int(b14[0][0][0]), int(b14[0][0][1]))

        if draw_rect1 ==True:
            cv2.line(copy,b1,b3,(255,255,0),10)
            cv2.line(copy,b3,b2,(255,255,0),10)
            cv2.line(copy,b2,b4,(255,255,0),10)
            cv2.line(copy,b4,b1,(255,255,0),10)

        if draw_rect2 ==True:
            cv2.line(copy,b11,b13,(255,255,0),10)
            cv2.line(copy,b13,b12,(255,255,0),10)
            cv2.line(copy,b12,b14,(255,255,0),10)
            cv2.line(copy,b14,b11,(255,255,0),10)

        if draw_lines == True:
            cv2.line(copy,b11,b1,(0,255,0),10)
            cv2.line(copy,b13,b3,(0,255,0),10)
            cv2.line(copy,b12,b2,(0,255,0),10)
            cv2.line(copy,b14,b4,(0,255,0),10)

        if draw_mask ==True:
            pt=np.float32([b11,b13,b14,b12])

            ty=add_photo(copy,pt,mask)
            tb= img_as_ubyte(ty)

            for i in range(0,ty.shape[0]):
                for j in range(0,ty.shape[1]):
                    k=ty[i,j]
                    if k[0] != -1 and k[1] != -1 and k[2] != -1:
                        copy[i,j] = tb[i,j]

    return copy

#change the photo
mask=cv2.imread("oni.jpg")

# the video
cap = cv2.VideoCapture("3.mp4")

if (cap.isOpened() == False):
    print("Unable to read camera feed")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while(True):
    ret, frame = cap.read()

    if ret == True:
        res=pro(frame,mask,draw_mask=True)
        cv2.imshow('head',res)
        # Write the frame into the file 'output.avi'
        out.write(res)

        # Break the loop
    else:
        break
    key = cv2.waitKey(10)
    if key == 27:
        cv2.destroyAllWindows()
        break

# When everything done, release the video capture and video write objects
cap.release()
out.release()

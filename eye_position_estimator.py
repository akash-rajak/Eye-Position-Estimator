
# Eye Position Estimator

# imported necessary library
from tkinter import *
import tkinter as tk
import tkinter.messagebox as mbox
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import matplotlib.pyplot as plt

import mediapipe as mp
import time
import numpy as np
from numpy import greater
import utils, math


# face bounder indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
# lips indices for Landmarks
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88,95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
# Left eyes indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
# right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]



# Main Window & Configuration
window = tk.Tk()
window.title("Eye Position Estimator")
window.iconbitmap('Images/icon.ico')
window.config(bg = "white")
window.geometry('1000x700')

# top label
start1 = tk.Label(text = "EYE POSITION\nESTIMATOR", font=("Arial", 45,"underline"), fg="magenta", bg = "white") # same way bg
start1.place(x = 280, y = 10)

# function defined to start the main application
def start_fun():
    window.destroy()

# created a start button
Button(window, text="▶ START",command=start_fun,font=("Arial", 25), bg = "orange", fg = "blue", cursor="hand2", borderwidth=3, relief="raised").place(x =130 , y =570 )

# image on the main window
path1 = "Images/eye.jpg"
img1 = ImageTk.PhotoImage(Image.open(path1))
panel1 = tk.Label(window, image = img1)
panel1.place(x = 200, y = 170)

exit1 = False
# function created for exiting from window
def exit_win():
    global exit1
    if mbox.askokcancel("Exit", "Do you want to exit?"):
        exit1 = True
        window.destroy()

# exit button created
Button(window, text="❌ EXIT",command=exit_win,font=("Arial", 25), bg = "red", fg = "blue", cursor="hand2", borderwidth=3, relief="raised").place(x =680 , y = 570 )

window.protocol("WM_DELETE_WINDOW", exit_win)
window.mainloop()

if exit1==False:
    # Main Window & Configuration of window1
    window1 = tk.Tk()
    window1.title("Eye Position Estimator")
    window1.iconbitmap('Images/icon.ico')
    window1.geometry('1000x700')

    filename=""
    filename1=""
    filename2=""

    # ---------------------------- video section ------------------------------------------------------------
    def video_option():
        # new windowv created for video section
        windowv = tk.Tk()
        windowv.title("Detection from Video")
        windowv.iconbitmap('Images/icon.ico')
        windowv.geometry('1000x700')


        # function defined to open the video
        def open_vid():
            global filename2

            filename2 = filedialog.askopenfilename(title="Select Video file", parent=windowv)
            path_text2.delete("1.0", "end")
            path_text2.insert(END, filename2)

        # function defined to detect inside the video
        def det_vid():
            global filename2
            b_list = []
            frame1 = []

            video_path = filename2
            if (video_path == ""):
                mbox.showerror("Error", "No Video File Selected!", parent = windowv)
                return
            info1.config(text="Status : Detecting...")
            mbox.showinfo("Status", "Detecting, Please Wait...", parent=windowv)
            # time.sleep(1)

            # camera object
            camera = cv2.VideoCapture(video_path)

            # variables
            frame_counter = 0
            CEF_COUNTER = 0
            TOTAL_BLINKS = 0
            # constants
            CLOSED_EYES_FRAME = 3
            FONTS = cv2.FONT_HERSHEY_COMPLEX

            # to get the face mesh points
            map_face_mesh = mp.solutions.face_mesh

            # landmark detection function
            def landmarksDetection(img, results, draw=False):
                img_height, img_width = img.shape[:2]
                # list[(x,y), (x,y)....]
                mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                              results.multi_face_landmarks[0].landmark]
                if draw:
                    [cv2.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

                # returning the list of tuples for each landmarks
                return mesh_coord

            # Euclaidean distance
            def euclaideanDistance(point, point1):
                x, y = point
                x1, y1 = point1
                distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
                return distance

            # Blinking Ratio
            def blinkRatio(img, landmarks, right_indices, left_indices):
                # Right eyes
                # horizontal line
                rh_right = landmarks[right_indices[0]]
                rh_left = landmarks[right_indices[8]]
                # vertical line
                rv_top = landmarks[right_indices[12]]
                rv_bottom = landmarks[right_indices[4]]
                # draw lines on right eyes
                # cv2.line(img, rh_right, rh_left, utils.GREEN, 2)
                # cv2.line(img, rv_top, rv_bottom, utils.WHITE, 2)

                # LEFT_EYE
                # horizontal line
                lh_right = landmarks[left_indices[0]]
                lh_left = landmarks[left_indices[8]]

                # vertical line
                lv_top = landmarks[left_indices[12]]
                lv_bottom = landmarks[left_indices[4]]

                rhDistance = euclaideanDistance(rh_right, rh_left)
                rvDistance = euclaideanDistance(rv_top, rv_bottom)

                lvDistance = euclaideanDistance(lv_top, lv_bottom)
                lhDistance = euclaideanDistance(lh_right, lh_left)

                reRatio = rhDistance / rvDistance
                leRatio = lhDistance / lvDistance

                ratio = (reRatio + leRatio) / 2
                return ratio

            # Eyes Extrctor function,
            def eyesExtractor(img, right_eye_coords, left_eye_coords):
                # converting color image to  scale image
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # getting the dimension of image
                dim = gray.shape
                # creating mask from gray scale dim
                mask = np.zeros(dim, dtype=np.uint8)
                # drawing Eyes Shape on mask with white color
                cv2.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
                cv2.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

                # showing the mask
                # cv2.imshow('mask', mask)

                # draw eyes image on mask, where white shape is
                eyes = cv2.bitwise_and(gray, gray, mask=mask)
                # change black color to gray other than eys
                # cv2.imshow('eyes draw', eyes)
                eyes[mask == 0] = 155

                # getting minium and maximum x and y  for right and left eyes
                # For Right Eye
                r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
                r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
                r_max_y = (max(right_eye_coords, key=lambda item: item[1]))[1]
                r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

                # For LEFT Eye
                l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
                l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
                l_max_y = (max(left_eye_coords, key=lambda item: item[1]))[1]
                l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

                # croping the eyes from mask
                cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
                cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

                # returning the cropped eyes
                return cropped_right, cropped_left

            # Eyes Postion Estimator
            def positionEstimator(cropped_eye):
                # getting height and width of eye
                h, w = cropped_eye.shape

                # remove the noise from images
                gaussain_blur = cv2.GaussianBlur(cropped_eye, (9, 9), 0)
                median_blur = cv2.medianBlur(gaussain_blur, 3)

                # applying thrsholding to convert binary_image
                ret, threshed_eye = cv2.threshold(median_blur, 130, 255, cv2.THRESH_BINARY)

                # create fixd part for eye with
                piece = int(w / 3)

                # slicing the eyes into three parts
                right_piece = threshed_eye[0:h, 0:piece]
                center_piece = threshed_eye[0:h, piece: piece + piece]
                left_piece = threshed_eye[0:h, piece + piece:w]

                # calling pixel counter function
                eye_position, color = pixelCounter(right_piece, center_piece, left_piece)

                return eye_position, color

            # creating pixel counter function
            def pixelCounter(first_piece, second_piece, third_piece):
                # counting black pixel in each part
                right_part = np.sum(first_piece == 0)
                center_part = np.sum(second_piece == 0)
                left_part = np.sum(third_piece == 0)
                # creating list of these values
                eye_parts = [right_part, center_part, left_part]

                # getting the index of max values in the list
                max_index = eye_parts.index(max(eye_parts))
                pos_eye = ''
                if max_index == 0:
                    pos_eye = "RIGHT"
                    color = [utils.BLACK, utils.GREEN]
                elif max_index == 1:
                    pos_eye = 'CENTER'
                    color = [utils.YELLOW, utils.PINK]
                elif max_index == 2:
                    pos_eye = 'LEFT'
                    color = [utils.GRAY, utils.YELLOW]
                else:
                    pos_eye = "Closed"
                    color = [utils.GRAY, utils.YELLOW]
                return pos_eye, color

            with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

                # starting time here
                start_time = time.time()
                # starting Video loop here.
                x = 0
                while True:
                    frame_counter += 1  # frame counter
                    ret, frame = camera.read()  # getting frame from camera
                    if not ret:
                        break  # no more frames break
                    #  resizing frame

                    frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                    frame_height, frame_width = frame.shape[:2]
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    results = face_mesh.process(rgb_frame)
                    x1 = 0
                    if results.multi_face_landmarks:
                        mesh_coords = landmarksDetection(frame, results, False)
                        ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
                        # cv2.putText(frame, f'ratio {ratio}', (100, 100), FONTS, 1.0, utils.GREEN, 2)
                        # utils.colorBackgroundText(frame, f'Ratio : {round(ratio, 2)}', FONTS, 0.7, (30, 100), 2,
                        #                           utils.PINK, utils.YELLOW)

                        if ratio > 3.2: #5.5
                            CEF_COUNTER += 1
                            # cv2.putText(frame, 'Blink', (200, 50), FONTS, 1.3, utils.PINK, 2)
                            utils.colorBackgroundText(frame, f'Blink', FONTS, 1.7, (int(frame_height / 2), 100), 2,
                                                      utils.BLUE, pad_x=30, pad_y=6, )

                        else:
                            if CEF_COUNTER > CLOSED_EYES_FRAME:
                                TOTAL_BLINKS += 1
                                CEF_COUNTER = 0
                                b_list.append(1)
                                x1 = 1
                        # cv2.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (100, 150), FONTS, 0.6, utils.BLUE, 2)
                        utils.colorBackgroundText(frame, f'Total Blinks: {TOTAL_BLINKS}', FONTS, 1.5, (30, 150), 2)

                        cv2.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True,
                                      utils.YELLOW, 3, cv2.LINE_AA)
                        cv2.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True,
                                      utils.YELLOW, 2, cv2.LINE_AA)

                        # Blink Detector Counter Completed
                        right_coords = [mesh_coords[p] for p in RIGHT_EYE]
                        left_coords = [mesh_coords[p] for p in LEFT_EYE]
                        crop_right, crop_left = eyesExtractor(frame, right_coords, left_coords)
                        # cv2.imshow('right', crop_right)
                        # cv2.imshow('left', crop_left)
                        eye_position, color = positionEstimator(crop_right)
                        utils.colorBackgroundText(frame, f'RIGHT : {eye_position}', FONTS, 1.5, (40, 220), 2, color[0],
                                                  color[1], 8, 8)
                        eye_position_left, color = positionEstimator(crop_left)
                        utils.colorBackgroundText(frame, f'LEFT : {eye_position_left}', FONTS, 1.5, (40, 320), 2, color[0],
                                                  color[1], 8, 8)

                    # calculating  frame per seconds FPS
                    end_time = time.time() - start_time
                    fps = frame_counter / end_time

                    # frame = utils.textWithBackground(frame, f'FPS: {round(fps, 1)}', FONTS, 1.0, (30, 50),
                    #                                  bgOpacity=0.9, textThickness=2)
                    # writing image for thumbnail drawing shape
                    # cv2.imwrite(f'img/frame_{frame_counter}.png', frame)

                    x += 1
                    frame1.append(x)
                    if (x1 == 0):
                        b_list.append(0)

                    frame = cv2.resize(frame, (1000, 650))
                    cv2.imshow('Eye Position Detection', frame)
                    key = cv2.waitKey(2)
                    if key == ord('q') or key == ord('Q'):
                        break
                cv2.destroyAllWindows()
                camera.release()

            info1.config(text="                                                  ")
            info1.config(text="Status : Detection Completed")
            cv2.destroyAllWindows()

            def graph_analysis():
                plt.figure(facecolor='orange', )
                ax = plt.axes()
                ax.set_facecolor("yellow")
                plt.plot(frame1, b_list, label="Eye Blink", color="green", marker='o', markerfacecolor='blue')
                plt.xlabel('Time (sec)')
                plt.ylabel('Eye Blink')
                plt.title('Eye Blink Plot')
                plt.legend()
                plt.get_current_fig_manager().canvas.set_window_title("Graph Analysis")
                plt.show()

            Button(windowv, text="Graph Analysis", command=graph_analysis, cursor="hand2", font=("Arial", 20), bg="orange", fg="blue").place(x=370, y=530)


        def land_vid():
            global filename2

            video_path = filename2
            if (video_path == ""):
                mbox.showerror("Error", "No Video File Selected!", parent=windowv)
                return

            # camera object
            camera = cv2.VideoCapture(video_path)

            # variables
            frame_counter = 0
            # constants
            FONTS = cv2.FONT_HERSHEY_COMPLEX

            map_face_mesh = mp.solutions.face_mesh

            # landmark detection function
            def landmarksDetection(img, results, draw=False):
                img_height, img_width = img.shape[:2]
                # list[(x,y), (x,y)....]
                mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
                if draw:
                    [cv2.circle(img, p, 2, utils.GREEN, -1) for p in mesh_coord]

                # returning the list of tuples for each landmarks
                return mesh_coord

            with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
                # starting time here
                start_time = time.time()
                # starting Video loop here.
                while True:
                    frame_counter += 1  # frame counter
                    ret, frame = camera.read()  # getting frame from camera
                    if not ret:
                        break  # no more frames break
                    #  resizing frame
                    # frame = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                    # writing orginal image image thumbnail
                    # cv2.imwrite(f'img/img_{frame_counter}.png', frame)
                    # print(frame_counter)

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    results = face_mesh.process(rgb_frame)
                    if results.multi_face_landmarks:
                        mesh_coords = landmarksDetection(frame, results, False)
                        frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in FACE_OVAL], utils.WHITE, opacity=0.4)
                        frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in LEFT_EYE], utils.GREEN, opacity=0.4)
                        frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in RIGHT_EYE], utils.GREEN, opacity=0.4)
                        frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in LEFT_EYEBROW], utils.ORANGE, opacity=0.4)
                        frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in RIGHT_EYEBROW], utils.ORANGE, opacity=0.4)
                        frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in LIPS], utils.BLACK, opacity=0.3)

                        [cv2.circle(frame, mesh_coords[p], 1, utils.GREEN, -1, cv2.LINE_AA) for p in LIPS]
                        [cv2.circle(frame, mesh_coords[p], 1, utils.BLACK, - 1, cv2.LINE_AA) for p in RIGHT_EYE]
                        [cv2.circle(frame, mesh_coords[p], 1, utils.BLACK, -1, cv2.LINE_AA) for p in LEFT_EYE]

                        [cv2.circle(frame, mesh_coords[p], 1, utils.BLACK, -1, cv2.LINE_AA) for p in RIGHT_EYEBROW]
                        [cv2.circle(frame, mesh_coords[p], 1, utils.BLACK, -1, cv2.LINE_AA) for p in LEFT_EYEBROW]
                        [cv2.circle(frame, mesh_coords[p], 1, utils.RED, -1, cv2.LINE_AA) for p in FACE_OVAL]

                    # calculating  frame per seconds FPS
                    end_time = time.time() - start_time
                    fps = frame_counter / end_time

                    # frame = utils.textWithBackground(frame, f'FPS: {round(fps, 1)}', FONTS, 1.0, (20, 50), bgOpacity=0.9, textThickness=2)
                    # writing image for thumbnail drawing shape
                    # cv2.imwrite(f'img/frame_{frame_counter}.png', frame)
                    frame = cv2.resize(frame, (1000, 650))
                    cv2.imshow('Landmarks From Video', frame)
                    key = cv2.waitKey(1)
                    if key == ord('q') or key == ord('Q'):
                        break
                cv2.destroyAllWindows()
                camera.release()


        # funcion defined to preview the selected video
        def prev_vid():
            global filename2
            cap = cv2.VideoCapture(filename2)
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    img = cv2.resize(frame, (800, 500))
                    cv2.imshow('Selected Video Preview', img)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                else:
                    break
            cap.release()
            cv2.destroyAllWindows()


        lbl1 = tk.Label(windowv, text="DETECT  FROM\nVIDEO", font=("Arial", 50, "underline"), fg="brown")
        lbl1.place(x=230, y=20)
        lbl2 = tk.Label(windowv, text="Selected Video", font=("Arial", 30), fg="green")
        lbl2.place(x=80, y=200)
        path_text2 = tk.Text(windowv, height=1, width=37, font=("Arial", 30), bg="light yellow", fg="orange", borderwidth=2,relief="solid")
        path_text2.place(x=80, y=260)

        Button(windowv, text="SELECT", command=open_vid, cursor="hand2", font=("Arial", 20), bg="light green", fg="blue").place(x=140, y=350)
        Button(windowv, text="PREVIEW", command=prev_vid, cursor="hand2", font=("Arial", 20), bg="yellow", fg="blue").place(x=300, y=350)
        Button(windowv, text="LANDMARKS", command=land_vid, cursor="hand2", font=("Arial", 20), bg="yellow",fg="blue").place(x=480, y=350)
        Button(windowv, text="DETECT", command=det_vid, cursor="hand2", font=("Arial", 20), bg="orange", fg="blue").place(x=710, y=350)

        info1 = tk.Label(windowv, font=("Arial", 30), fg="gray")  # same way bg
        info1.place(x=100, y=440)

        #function defined to exit from windowv section
        def exit_winv():
            if mbox.askokcancel("Exit", "Do you want to exit?", parent = windowv):
                windowv.destroy()
        windowv.protocol("WM_DELETE_WINDOW", exit_winv)



    # ---------------------------- camera section ------------------------------------------------------------
    def camera_option():
        # new window created for camera section
        windowc = tk.Tk()
        windowc.title("Human Detection from Camera")
        windowc.iconbitmap('Images/icon.ico')
        windowc.geometry('1000x700')

        def open_cam_land():
            # camera object
            camera = cv2.VideoCapture(0)

            # variables
            frame_counter = 0
            # constants
            FONTS = cv2.FONT_HERSHEY_COMPLEX

            map_face_mesh = mp.solutions.face_mesh

            # landmark detection function
            def landmarksDetection(img, results, draw=False):
                img_height, img_width = img.shape[:2]
                # list[(x,y), (x,y)....]
                mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                              results.multi_face_landmarks[0].landmark]
                if draw:
                    [cv2.circle(img, p, 2, utils.GREEN, -1) for p in mesh_coord]

                # returning the list of tuples for each landmarks
                return mesh_coord

            with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
                # starting time here
                start_time = time.time()
                # starting Video loop here.
                while True:
                    frame_counter += 1  # frame counter
                    ret, frame = camera.read()  # getting frame from camera
                    if not ret:
                        break  # no more frames break
                    #  resizing frame
                    # frame = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                    # writing orginal image image thumbnail
                    # cv2.imwrite(f'img/img_{frame_counter}.png', frame)
                    # print(frame_counter)

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    results = face_mesh.process(rgb_frame)
                    if results.multi_face_landmarks:
                        mesh_coords = landmarksDetection(frame, results, False)
                        frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in FACE_OVAL], utils.WHITE,
                                                    opacity=0.4)
                        frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in LEFT_EYE], utils.GREEN, opacity=0.4)
                        frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in RIGHT_EYE], utils.GREEN,
                                                    opacity=0.4)
                        frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in LEFT_EYEBROW], utils.ORANGE,
                                                    opacity=0.4)
                        frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in RIGHT_EYEBROW], utils.ORANGE,
                                                    opacity=0.4)
                        frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in LIPS], utils.BLACK, opacity=0.3)

                        [cv2.circle(frame, mesh_coords[p], 1, utils.GREEN, -1, cv2.LINE_AA) for p in LIPS]
                        [cv2.circle(frame, mesh_coords[p], 1, utils.BLACK, - 1, cv2.LINE_AA) for p in RIGHT_EYE]
                        [cv2.circle(frame, mesh_coords[p], 1, utils.BLACK, -1, cv2.LINE_AA) for p in LEFT_EYE]

                        [cv2.circle(frame, mesh_coords[p], 1, utils.BLACK, -1, cv2.LINE_AA) for p in RIGHT_EYEBROW]
                        [cv2.circle(frame, mesh_coords[p], 1, utils.BLACK, -1, cv2.LINE_AA) for p in LEFT_EYEBROW]
                        [cv2.circle(frame, mesh_coords[p], 1, utils.RED, -1, cv2.LINE_AA) for p in FACE_OVAL]

                    # calculating  frame per seconds FPS
                    end_time = time.time() - start_time
                    fps = frame_counter / end_time

                    # frame = utils.textWithBackground(frame, f'FPS: {round(fps, 1)}', FONTS, 1.0, (20, 50), bgOpacity=0.9, textThickness=2)
                    # writing image for thumbnail drawing shape
                    # cv2.imwrite(f'img/frame_{frame_counter}.png', frame)
                    frame = cv2.resize(frame, (1000, 650))
                    cv2.imshow('Landmarks From Video', frame)
                    key = cv2.waitKey(1)
                    if key == ord('q') or key == ord('Q'):
                        break
                cv2.destroyAllWindows()
                camera.release()

        # function defined to open the camera
        def open_cam_detect():
            b_list = []
            frame1 = []

            info1.config(text="Status : Opening Camera...")
            mbox.showinfo("Status", "Opening Camera...Please Wait...", parent=windowc)
            # time.sleep(1)

            # camera object
            camera = cv2.VideoCapture(0)

            # variables
            frame_counter = 0
            CEF_COUNTER = 0
            TOTAL_BLINKS = 0
            # constants
            CLOSED_EYES_FRAME = 3
            FONTS = cv2.FONT_HERSHEY_COMPLEX

            map_face_mesh = mp.solutions.face_mesh

            # landmark detection function
            def landmarksDetection(img, results, draw=False):
                img_height, img_width = img.shape[:2]
                # list[(x,y), (x,y)....]
                mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                              results.multi_face_landmarks[0].landmark]
                if draw:
                    [cv2.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

                # returning the list of tuples for each landmarks
                return mesh_coord

            # Euclaidean distance
            def euclaideanDistance(point, point1):
                x, y = point
                x1, y1 = point1
                distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
                return distance

            # Blinking Ratio
            def blinkRatio(img, landmarks, right_indices, left_indices):
                # Right eyes
                # horizontal line
                rh_right = landmarks[right_indices[0]]
                rh_left = landmarks[right_indices[8]]
                # vertical line
                rv_top = landmarks[right_indices[12]]
                rv_bottom = landmarks[right_indices[4]]
                # draw lines on right eyes
                # cv2.line(img, rh_right, rh_left, utils.GREEN, 2)
                # cv2.line(img, rv_top, rv_bottom, utils.WHITE, 2)

                # LEFT_EYE
                # horizontal line
                lh_right = landmarks[left_indices[0]]
                lh_left = landmarks[left_indices[8]]

                # vertical line
                lv_top = landmarks[left_indices[12]]
                lv_bottom = landmarks[left_indices[4]]

                rhDistance = euclaideanDistance(rh_right, rh_left)
                rvDistance = euclaideanDistance(rv_top, rv_bottom)

                lvDistance = euclaideanDistance(lv_top, lv_bottom)
                lhDistance = euclaideanDistance(lh_right, lh_left)

                reRatio = rhDistance / rvDistance
                leRatio = lhDistance / lvDistance

                ratio = (reRatio + leRatio) / 2
                return ratio

            # Eyes Extrctor function,
            def eyesExtractor(img, right_eye_coords, left_eye_coords):
                # converting color image to  scale image
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # getting the dimension of image
                dim = gray.shape
                # creating mask from gray scale dim
                mask = np.zeros(dim, dtype=np.uint8)
                # drawing Eyes Shape on mask with white color
                cv2.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
                cv2.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

                # showing the mask
                # cv2.imshow('mask', mask)

                # draw eyes image on mask, where white shape is
                eyes = cv2.bitwise_and(gray, gray, mask=mask)
                # change black color to gray other than eys
                # cv2.imshow('eyes draw', eyes)
                eyes[mask == 0] = 155

                # getting minium and maximum x and y  for right and left eyes
                # For Right Eye
                r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
                r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
                r_max_y = (max(right_eye_coords, key=lambda item: item[1]))[1]
                r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

                # For LEFT Eye
                l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
                l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
                l_max_y = (max(left_eye_coords, key=lambda item: item[1]))[1]
                l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

                # croping the eyes from mask
                cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
                cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

                # returning the cropped eyes
                return cropped_right, cropped_left

            # Eyes Postion Estimator
            def positionEstimator(cropped_eye):
                # getting height and width of eye
                h, w = cropped_eye.shape

                # remove the noise from images
                gaussain_blur = cv2.GaussianBlur(cropped_eye, (9, 9), 0)
                median_blur = cv2.medianBlur(gaussain_blur, 3)

                # applying thrsholding to convert binary_image
                ret, threshed_eye = cv2.threshold(median_blur, 130, 255, cv2.THRESH_BINARY)

                # create fixd part for eye with
                piece = int(w / 3)

                # slicing the eyes into three parts
                right_piece = threshed_eye[0:h, 0:piece]
                center_piece = threshed_eye[0:h, piece: piece + piece]
                left_piece = threshed_eye[0:h, piece + piece:w]

                # calling pixel counter function
                eye_position, color = pixelCounter(right_piece, center_piece, left_piece)

                return eye_position, color

            # creating pixel counter function
            def pixelCounter(first_piece, second_piece, third_piece):
                # counting black pixel in each part
                right_part = np.sum(first_piece == 0)
                center_part = np.sum(second_piece == 0)
                left_part = np.sum(third_piece == 0)
                # creating list of these values
                eye_parts = [right_part, center_part, left_part]

                # getting the index of max values in the list
                max_index = eye_parts.index(max(eye_parts))
                pos_eye = ''
                if max_index == 0:
                    pos_eye = "RIGHT"
                    color = [utils.BLACK, utils.GREEN]
                elif max_index == 1:
                    pos_eye = 'CENTER'
                    color = [utils.YELLOW, utils.PINK]
                elif max_index == 2:
                    pos_eye = 'LEFT'
                    color = [utils.GRAY, utils.YELLOW]
                else:
                    pos_eye = "Closed"
                    color = [utils.GRAY, utils.YELLOW]
                return pos_eye, color

            with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

                # starting time here
                start_time = time.time()
                # starting Video loop here.
                x = 0
                while True:
                    frame_counter += 1  # frame counter
                    ret, frame = camera.read()  # getting frame from camera
                    if not ret:
                        break  # no more frames break
                    #  resizing frame

                    frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                    frame_height, frame_width = frame.shape[:2]
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    results = face_mesh.process(rgb_frame)
                    x1 = 0
                    if results.multi_face_landmarks:
                        mesh_coords = landmarksDetection(frame, results, False)
                        ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
                        # cv2.putText(frame, f'ratio {ratio}', (100, 100), FONTS, 1.0, utils.GREEN, 2)
                        # utils.colorBackgroundText(frame, f'Ratio : {round(ratio, 2)}', FONTS, 0.7, (30, 100), 2,
                        #                           utils.PINK, utils.YELLOW)

                        if ratio > 4.0: #5.5
                            CEF_COUNTER += 1
                            # cv2.putText(frame, 'Blink', (200, 50), FONTS, 1.3, utils.PINK, 2)
                            utils.colorBackgroundText(frame, f'Blink', FONTS, 1.7, (int(frame_height / 2), 100), 2,
                                                      utils.BLUE, pad_x=6, pad_y=6, )

                        else:
                            if CEF_COUNTER > CLOSED_EYES_FRAME:
                                TOTAL_BLINKS += 1
                                CEF_COUNTER = 0
                                b_list.append(1)
                                x1 = 1
                        # cv2.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (100, 150), FONTS, 0.6, utils.GREEN, 2)
                        utils.colorBackgroundText(frame, f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30, 150), 2)

                        cv2.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True,
                                     utils.YELLOW, 2, cv2.LINE_AA)
                        cv2.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True,
                                     utils.YELLOW, 2, cv2.LINE_AA)

                        # Blink Detector Counter Completed
                        right_coords = [mesh_coords[p] for p in RIGHT_EYE]
                        left_coords = [mesh_coords[p] for p in LEFT_EYE]
                        crop_right, crop_left = eyesExtractor(frame, right_coords, left_coords)
                        # cv2.imshow('right', crop_right)
                        # cv2.imshow('left', crop_left)
                        eye_position, color = positionEstimator(crop_right)
                        utils.colorBackgroundText(frame, f'RIGHT : {eye_position}', FONTS, 1.0, (40, 220), 2, color[0],
                                                  color[1], 8, 8)
                        eye_position_left, color = positionEstimator(crop_left)
                        utils.colorBackgroundText(frame, f'LEFT : {eye_position_left}', FONTS, 1.0, (40, 320), 2, color[0],
                                                  color[1], 8, 8)

                    # calculating  frame per seconds FPS
                    end_time = time.time() - start_time
                    fps = frame_counter / end_time

                    # frame = utils.textWithBackground(frame, f'FPS: {round(fps, 1)}', FONTS, 1.0, (30, 50),
                    #                                  bgOpacity=0.9, textThickness=2)
                    # writing image for thumbnail drawing shape
                    # cv2.imwrite(f'img/frame_{frame_counter}.png', frame)
                    x += 1
                    frame1.append(x)
                    if(x1==0):
                        b_list.append(0)
                    cv2.imshow('Eye Position Detection', frame)
                    key = cv2.waitKey(2)
                    if key == ord('q') or key == ord('Q'):
                        break
                cv2.destroyAllWindows()
                camera.release()


            info1.config(text="                                                  ")
            info1.config(text="Status : Detection Completed")
            cv2.destroyAllWindows()

            def graph_analysis():
                plt.figure(facecolor='orange', )
                ax = plt.axes()
                ax.set_facecolor("yellow")
                plt.plot(frame1, b_list, label="Eye Blink", color="green", marker='o', markerfacecolor='blue')
                plt.xlabel('Time (sec)')
                plt.ylabel('Eye Blink')
                plt.title('Eye Blink Plot')
                plt.legend()
                plt.get_current_fig_manager().canvas.set_window_title("Graph Analysis")
                plt.show()

            Button(windowc, text="Graph Analysis", command=graph_analysis, cursor="hand2", font=("Arial", 20), bg="orange", fg="blue").place(x=370, y=500)

        lbl1 = tk.Label(windowc, text="DETECT  FROM\nCAMERA", font=("Arial", 50, "underline"), fg="brown")  # same way bg
        lbl1.place(x=230, y=20)

        Label(windowc, text="LANDMARKS", font=("Arial", 25, "bold"), fg="green").place(x=165, y=220)
        Button(windowc, text="OPEN CAMERA", command=open_cam_land, cursor="hand2", font=("Arial", 20), bg="light green", fg="blue").place(x=160, y=280)
        Label(windowc, text="DETECT", font=("Arial", 25, "bold"), fg="green").place(x=615, y=220)
        Button(windowc, text="OPEN CAMERA", command=open_cam_detect, cursor="hand2", font=("Arial", 20), bg="light green",fg="blue").place(x=570, y=280)

        info1 = tk.Label(windowc, font=("Arial", 30), fg="gray")  # same way bg
        info1.place(x=100, y=380)

        # function defined to exit from the camera window
        def exit_winc():
            if mbox.askokcancel("Exit", "Do you want to exit?", parent = windowc):
                windowc.destroy()
        windowc.protocol("WM_DELETE_WINDOW", exit_winc)


    # options -----------------------------
    lbl1 = tk.Label(text="OPTIONS", font=("Arial", 50, "underline"),fg="brown")  # same way bg
    lbl1.place(x=340, y=20)

    # image on the main window
    pathv = "Images/image2.png"
    imgv = ImageTk.PhotoImage(Image.open(pathv))
    panelv = tk.Label(window1, image = imgv)
    panelv.place(x = 700, y = 160)# 720, 260

    # image on the main window
    pathc = "Images/image3.jpg"
    imgc = ImageTk.PhotoImage(Image.open(pathc))
    panelc = tk.Label(window1, image = imgc)
    panelc.place(x = 90, y = 365)

    # created button for all three option
    Button(window1, text="DETECT  FROM  VIDEO ➡",command=video_option, cursor="hand2", font=("Arial", 30), bg = "light blue", fg = "blue").place(x = 110, y = 200) #90, 300
    Button(window1, text="DETECT FROM CAMERA ➡",command=camera_option, cursor="hand2", font=("Arial", 30), bg = "light green", fg = "blue").place(x = 350, y = 400)

    # function defined to exit from window1
    def exit_win1():
        if mbox.askokcancel("Exit", "Do you want to exit?"):
            window1.destroy()

    # created exit button
    Button(window1, text="❌ EXIT",command=exit_win1,  cursor="hand2", font=("Arial", 25), bg = "red", fg = "blue").place(x = 440, y = 600)

    window1.protocol("WM_DELETE_WINDOW", exit_win1)
    window1.mainloop()


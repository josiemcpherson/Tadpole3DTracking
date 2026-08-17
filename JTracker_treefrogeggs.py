import cv2
import os
import numpy as np
from collections import deque
import Tracking_functions1 as tc
#Importing all the packages needed and the tracking functions



# Information to change
sub_varThreshold = 90 #values usually work between 90-150 - the pixel threshold value to be counted as a "moving tadpole" object
sub_learn_rate = -1 # -1 is the default, 0.05 also works well
blur_kernal = (1, 1) #use smaller kernal sizes (1,1) for small tadpoles and larger kernals (5,5) for big tadpoles
min_area = 1 #initalise min area as a number before setting by drawing a rectangle
max_area = 1 #initalise max area as a number before setting by drawing a rectangle
large_movement_threshold = 1 #initalise threshold as a number before setting by drawing a rectangle
start_position = 2 #where we start
start_frame = start_position  #the videos are fps so if start position is in seconds, this takes us to frames
min_area_multiplier = 0.4 #the min area we will track will be 0.4 of the rectangle drawn around the individual
max_area_multiplier = 1.3 #the max area we will track will be 0.4 of the rectangle drawn around the individual
key = None #initalise 
cap = None #initalise
frame_count = 0  # Variable to keep track of the frame number





# Create dictionaries to store window statuses
windows = {
    'Draw Tracking Arena': False,
    'Draw Area to Track': False
}

# Create a function to create and show windows
def create_window(window_name, image):
    cv2.imshow(window_name, image)
    windows[window_name] = True

# Create a function to destroy windows
def destroy_window(window_name):
    cv2.destroyWindow(window_name)
    windows[window_name] = False

# Create one numpy arrays to place over images, I am creating ROI here because I seem to need to do that to clear ROI between batches
img = np.zeros((360, 640, 3), np.uint8)
ROI = np.zeros_like(img)

# Open and read the txt file of video names
myfile = open('C:/Users/s2250312/Documents/phd/treefrogeggs/TRACK1.txt', encoding="ISO-8859-1")
myfile = myfile.readlines()
batch_numbers = set()
 

for line in myfile[1:]:
    line = line.strip()
    line = line.split()
    BatchID = 0
    TadpoleID = 1
    TrialID = 2
    columnvid = 3
    LR = 4
    batch_numbers.add(line[BatchID])

for batch in batch_numbers:
    ROI.fill(0) #clearing ROI between batches
    print("Is ROI cleared:", np.count_nonzero(ROI) == 0)
    print("ROI dimensions (width, height):", ROI.shape)
    

    top_left_pt1 = (0, 0) #resetting tracking area and arena
    bottom_right_pt1 = (0, 0)
    top_left_pt2 = (0, 0)
    bottom_right_pt2 = (0, 0)
        

   

    for line in myfile[1:]:
        line = line.strip()
        line = line.split()
        BatchID = 0
        if line[BatchID] != batch:
         continue
        TadpoleID = 1
        TrialID = 2
        columnvid = 3
        LR = 4
        Behav = 5
        vidx = (line[columnvid])

           # Create a unique window name for each tadpole based on TadpoleID
        tracking_window_name = f'Tracking Arena for Individual {line[TadpoleID]} on the {line[LR]} {line[Behav]}'
        area_window_name = f'Area of Individual {line[TadpoleID]} to Track on the {line[LR]}  {line[Behav]}'
        results_fn = ('_'.join(('Track_Results/Batch', batch, 'Individial', line[TadpoleID], 'Trial', line[TrialID], 'Behav', line[Behav], line[LR])))
   
        suffix = '.txt'
        results_txt = os.path.join(results_fn + suffix)
        myresults = open(results_txt, 'a')
        printheader = print('TadpoleID', 'TrialID', 'FrameNo', 'Xcentroid', 'Ycentroid', 'DistX', 'DistY', 'PixelDist', 'CumulPixelDistTrvalled', 'CumulPixelDistInculded', 'InculsionStatus', sep='\t', file=myresults)
        myresults.close()
        myresults = open(results_txt, 'a')

        



        # Capture the first frame of the video
        
        cap = cv2.VideoCapture(vidx)

        

       

                

                
        

        ret, first_frame = cap.read()
        if not ret:
          print("Error: Failed to read the first frame")
          break


      
       
        
        square_image1 = first_frame.copy()
        create_window(tracking_window_name, square_image1)
        print("Window 'Draw Tracking Arena' opened.")

        def draw_square1(event, x, y, flags, param):
            global square_image1, drawing1, top_left_pt1, bottom_right_pt1
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing1 = True
                top_left_pt1 = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing1 = False
                bottom_right_pt1 = (x, y)
                cv2.rectangle(square_image1, top_left_pt1, bottom_right_pt1, (255, 255, 255), -1)

        drawing1 = False
        cv2.setMouseCallback(tracking_window_name, draw_square1)
        # Capture the first frame of the video
        cap = cv2.VideoCapture(vidx)

        # Skip the first 49 frames to reach the 50th frame
        for i in range(49):
         ret, _ = cap.read()
         if not ret:
          print("Error: Failed to read frame ", i)
          break

# Read the 50th frame for square_image2
        ret, square_image2 = cap.read()
        if not ret:
         print("Error: Failed to read the 50th frame for square_image2")
         break
        
        create_window(area_window_name, square_image2)
        print("Window 'Draw Area' opened.")

        def draw_square2(event, x, y, flags, param):
            global square_image2, drawing2, top_left_pt2, bottom_right_pt2
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing2 = True
                top_left_pt2 = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing2 = False
                bottom_right_pt2 = (x, y)
                cv2.rectangle(square_image2, top_left_pt2, bottom_right_pt2, (255, 255, 255), -1)

        drawing2 = False
        cv2.setMouseCallback(area_window_name, draw_square2)
        def check_for_keys():
          key = cv2.waitKey(1) & 0xFF
          return key
        
        while True:
            cv2.imshow(tracking_window_name, square_image1)
            cv2.imshow(area_window_name, square_image2)
        
            key = check_for_keys()

            if key == ord('q'):
              break
             
            elif key == ord('s'):
               break

           
            if top_left_pt1[0] < bottom_right_pt1[0] and top_left_pt1[1] < bottom_right_pt1[1]:
                ROI = cv2.rectangle(img, top_left_pt1, bottom_right_pt1, (255, 255, 255), -1)
            
            
           
                
            
            if top_left_pt2[0] < bottom_right_pt2[0] and top_left_pt2[1] < bottom_right_pt2[1]:
                drawn_area = abs(bottom_right_pt2[0] - top_left_pt2[0]) * abs(bottom_right_pt2[1] - top_left_pt2[1])
                width = abs(bottom_right_pt2[0] - top_left_pt2[0])
                height = abs(bottom_right_pt2[1] - top_left_pt2[1])
                longest_side_length = max(width, height)
                
                min_area = drawn_area * min_area_multiplier
                max_area = drawn_area * max_area_multiplier
                large_movement_threshold = 3.5 * longest_side_length
                print(large_movement_threshold)
            
                

        prev_x = deque(maxlen=1)
        prev_y = deque(maxlen=1)
        prev_xy = deque(maxlen=1)
        prev_x.append(0)
        prev_y.append(0)
        prev_xy.append(None)

        pts = deque(maxlen=100)
        cxpts = deque(maxlen=2)
        cypts = deque(maxlen=2)
        dist_travelled = deque()

        cap = cv2.VideoCapture(vidx)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.set(1, 1)
        end_frame = int(700 * fps)
        framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sub_history = 100
        subtractor = cv2.createBackgroundSubtractorMOG2(history=sub_history, varThreshold=sub_varThreshold, detectShadows=False)

        while True:
            frame = cap.read()
            frame = frame[1]
            frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

            if frame_pos > end_frame:
                myresults.close()
                break
            if frame is None:
                myresults.close()
                break

            frame = cv2.bitwise_and(frame, ROI)
            
            mask, blur, eq = tc.create_mask(frame, subtractor, sub_learn_rate, blur_kernal)
            c, cx, cy, cxcy = tc.detect_contours(frame, blur, mask, eq, min_area, max_area, prev_x, prev_y, prev_xy)
            print(cx,  cy)
            ix, iy, Pixel_dist, cumul_dist_travelled, cumul_dist_included, inclusion_status = tc.calculate_distance(cx, cy, cxpts, cypts, dist_travelled, large_movement_threshold)

           
                
            pts = tc.draw_lines(cxcy, pts, frame, blur, mask)

            printresults = print(line[TadpoleID], line[TrialID], int(frame_pos), cx, cy, ix, iy, Pixel_dist, cumul_dist_travelled, cumul_dist_included, inclusion_status, sep='\t', file=myresults) #print results to file

           

            
            frame, blur, mask, eq = tc.HUD_info(frame, blur, mask, eq, line, TadpoleID, frame_pos, cumul_dist_travelled, inclusion_status)

            output = tc.output(frame, mask, blur, eq, mode="frame+new_mask")

            key = tc.frame_display_time(fps, mode="veryfast_speed")

            cv2.imshow('tadpole_tracker', frame)
           
           

        

            if key == ord('q'):
                cv2.destroyAllWindows()
                break

        

cap.release()
cv2.destroyAllWindows()
myresults.close()

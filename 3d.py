import cv2
import time
import operator
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

#initialize various images
img = cv2.imread('Downloads/bg.jpg')
img2 = cv2.imread('Downloads/caution.jpeg')
img3 = cv2.imread('Downloads/drum.jpg')
img4 = cv2.imread('Downloads/drumblue.jpg')

#make copies
copy = img.copy()
copy_caution = img2.copy()
copy_drum = img3.copy()
copy_drum_blue = img4.copy()

#resize all copies
dim_copy2 = (280, 340)
dim_copy3 = (320, 400)
dim_copy4 = (240, 320)

copy_caution_new = cv2.resize(copy_caution, dim_copy2, interpolation = cv2.INTER_AREA)
copy_drum_new = cv2.resize(copy_drum, dim_copy3, interpolation = cv2.INTER_AREA)
copy_drum_blue_new = cv2.resize(copy_drum_blue, dim_copy4, interpolation = cv2.INTER_AREA)

#copy_boat_new = cv2.flip(copy_boat_new,1)

print(img.shape)
print(copy_caution_new.shape)
#tracker = cv2.TrackerKCF_create()
tracker = cv2.TrackerMedianFlow_create()
#tracker = cv2.TrackerTLD_create()
#tracker = cv2.TrackerBoosting_create()
#tracker = cv2.TrackerMIL_create()

tracker_name = str(tracker).split()[0][1:]

# Read video
cap = cv2.VideoCapture(0)

time.sleep(1)
# Read first frame.
ret, frame = cap.read()
#frame = cv2.flip(frame,1)

# Special function allows us to draw on the very first frame our desired ROI
roi = cv2.selectROI(frame, False)

(x,y,w,h) = tuple(map(int,roi))
p_initial = (x+w/2,y+h/2)
p3_initial = (w,h)

# Initialize tracker with first frame and bounding box
ret = tracker.init(frame, roi)

while True:
    # Read a new frame
    ret, frame = cap.read()
    #frame = cv2.flip(frame,1)
    
    
    # Update tracker
    success, roi = tracker.update(frame)
    
    # roi variable is a tuple of 4 floats
    # We need each value and we need them as integers
    (x,y,w,h) = tuple(map(int,roi))
    
    # Draw Rectangle as Tracker moves
    if success:
        # Tracking success
        p1 = (x, y)
        p2 = (x+w, y+h)
        cv2.rectangle(frame, p1, p2, (100,25,0), 3)
        #print(p1)
        #print(p2)
        p3 = (w,h)
        

        p_mid = (x+w/2,y+h/2)
        d_moved = tuple(map(operator.sub,p_initial ,p_mid ))
        p3_moved = tuple(map(operator.sub,p3_initial ,p3 ))

        u = int(p_mid[0])
        v = int(p_mid[1])

        #img[1000:1310,565:1065] = img2
        cropped = copy[v+500:v+1300, u+200:u+1640]
        #print(u)
        #print(v)


        #convert resized copies to gray 
        img2gray2 = cv2.cvtColor(copy_caution_new,cv2.COLOR_BGR2GRAY)
        img2gray3 = cv2.cvtColor(copy_drum_new,cv2.COLOR_BGR2GRAY)
        img2gray4 = cv2.cvtColor(copy_drum_blue_new,cv2.COLOR_BGR2GRAY)

        #cv2.imshow("gray",img2gray)

        #create masks for all copies
        ret,mask2 = cv2.threshold(img2gray2,220,255,cv2.THRESH_BINARY)
        ret,mask3 = cv2.threshold(img2gray3,220,255,cv2.THRESH_BINARY)
        ret,mask4 = cv2.threshold(img2gray4,220,255,cv2.THRESH_BINARY)

        #masking and assigning for blue drum
        u4 = int(u/4)
        v4 = int(v/4)
        #print(u4)
        #print(v4)
        roi4 = cropped[-v4+300:-v4+620,-u4+885:-u4+1125]
        roi_masked4 = cv2.bitwise_or(roi4, roi4, mask=mask4)
        mask_inv4 = cv2.bitwise_not(mask4)
        fg4 = cv2.bitwise_or(copy_drum_blue_new, copy_drum_blue_new, mask=mask_inv4)
        final_roi4 = cv2.bitwise_or(roi_masked4,fg4)
        cropped[-v4+300:-v4+620,-u4+885:-u4+1125] = final_roi4

        #masking and assigning for caution
        roi2 = cropped[325:665,745:1025]
        roi_masked2 = cv2.bitwise_or(roi2, roi2, mask=mask2)
        mask_inv2 = cv2.bitwise_not(mask2)
        fg2 = cv2.bitwise_or(copy_caution_new, copy_caution_new, mask=mask_inv2)
        final_roi2 = cv2.bitwise_or(roi_masked2,fg2)
        cropped[325:665,745:1025] = final_roi2

        #masking and assigning for red oil drum
        u3 = int(u/2)
        v3 = int(v/2)
        roi3 = cropped[v3+200:v3+600,u3+400:u3+720]
        roi_masked3 = cv2.bitwise_or(roi3, roi3, mask=mask3)
        mask_inv3 = cv2.bitwise_not(mask3)
        fg3 = cv2.bitwise_or(copy_drum_new, copy_drum_new, mask=mask_inv3)
        final_roi3 = cv2.bitwise_or(roi_masked3,fg3)
        cropped[v3+200:v3+600,u3+400:u3+720] = final_roi3

        #zooming and scaling
        scale_percent = 50 + p3_moved[0]/20 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        resized = cv2.resize(cropped, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("resized", resized)
       
        #resetting
        copy = img.copy()
        


    else :
        # Tracking failure
        cv2.putText(frame, "Failed to Detect Tracking", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)

    # Display tracker type on frame
    #cv2.putText(frame, tracker_name, (20,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3);

    # Display result
    cv2.imshow(tracker_name, frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : 
        break
        
cap.release()
cv2.destroyAllWindows()
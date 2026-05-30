from ultralytics import YOLO
import cv2
from sort.sort import *
from util import *
results={}
mot_tracker = Sort()

coco_model = YOLO('yolov8n.pt')#detects car
license_plate_detector = YOLO(r'C:\Users\AYUDH\Desktop\Computer_Vision\LicensePlate_Detection\LicensePlateDetectorModel_weights\best.pt')#detects license plates that I trained in kaggle
cap = cv2.VideoCapture(r'C:\Users\AYUDH\Desktop\Computer_Vision\LicensePlate_Detection\video\sample.mp4')
ret = True
frame_nmr = -1
vehicles = [2,3,5,7]
while ret:
    frame_nmr+=1
    ret, frame = cap.read()
    if ret:
        # if frame_nmr>10:
        #      break
        results[frame_nmr]={}
        detections = coco_model(frame)[0]
        detections_=[]
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1,y1,x2,y2,score])
        

#track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))      
#detect license plates
        license_plates = license_plate_detector(frame)[0]
        print("license plates:", license_plates.boxes.data.tolist())
        for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                #assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
                if car_id!=-1:
                    #crop license plate
                    license_plate_crop = frame[int(y1):int(y2),int(x1):int(x2),:]
                    #processing the license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray,64,255,cv2.THRESH_BINARY_INV)
                    #read license plate number
                    # license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray)
                    if license_plate_text is not None:
                         results[frame_nmr][car_id]={'car':{'bbox':[xcar1, ycar1, xcar2, ycar2]},
                                                     'license_plate':{'bbox':[x1, y1, x2, y2],
                                                                      'text':license_plate_text,
                                                                      'bbox_score':score,
                                                                      'text_score':license_plate_text_score}}
#results
write_csv(results, r'C:\Users\AYUDH\Desktop\Computer_Vision\LicensePlate_Detection\test.csv')

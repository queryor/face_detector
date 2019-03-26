#!/usr/bin/env python
# coding=utf-8

import torch
from association_lstm import build_association_lstm
import face_recognition
import argparse
import cv2
import time
import os
parser = argparse.ArgumentParser(
    description='S3FD face Detector Training With Pytorch')
parser.add_argument('--model',default='./weights/sfd_face.pth',help="model path")
parser.add_argument('--video',default='../face_tracking/data/test.mp4',help="video path")
args = parser.parse_args()
if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    model = build_association_lstm("test")
    model.load_weights(args.model)
    img_s = cv2.imread('10.jpg')
    h,w,c = img_s.shape
    img = cv2.resize(img_s,(300,300))
    img1 = torch.from_numpy(img).permute(2, 0, 1)
    input = img1.view(1,3,300,300).float()
    model = model.cuda(0)
    input = input.cuda(0)
    
 
    know_face_encodings = [
    ]
    known_face_names = [
    ]
    face_dir = './face_dataset/'
    for root,dir,files in os.walk(face_dir):
        for file in files:
            img_path = os.path.join(root,file)
            name = root.split('/')[-1]
            person = face_recognition.load_image_file(img_path)
            
            try:
                person_encoding = face_recognition.face_encodings(person)[0]
            except IndexError as e:
                print("{} encoding fault".format(img_path))
                continue
            know_face_encodings.append(person_encoding)
            known_face_names.append(name)


    # warm up 
    for i in range(2):
        start = time.time()
        output = model(input)
    cap = cv2.VideoCapture(args.video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w,h = cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    #print(w,h)
    w,h = int(w),int(h)
    video = cv2.VideoWriter('./output.mp4', fourcc, fps, (w,h))
    while True:
        ret,frame = cap.read()
        start = time.time()
        if not ret:
            break
        h,w,c = frame.shape
        img = cv2.resize(frame,(300,300))
        img1 = torch.from_numpy(img).permute(2, 0, 1)
        input = img1.view(1,3,300,300).float()
        input = input.cuda(0)
        
        output = model(input) 
        face_locations = [] 
        for b in output:
            for n,s in enumerate(b):
                if (s[1])>0.5:
                    x1,y1,x2,y2 = [x for x in s[2:]]
                    x1,x2 = int(x1*w),int(x2*w)
                    y1,y2 = int(y1*h),int(y2*h)
                    #cv2.imwrite('person{}.jpg'.format(n),frame[y1:y2,x1:x2])
                    face_locations.append((y1,x2,y2,x1))
                    #cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
        #print(face_locations) 
        face_encodings = face_recognition.face_encodings(frame[:,:,::-1],face_locations)
        #print(len(face_encodings))
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(know_face_encodings,face_encoding,0.5)
            #print(matches)
            name = "UnKnown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            face_names.append(name)
        for (top,right,bottom,left),name in zip(face_locations,face_names):
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2) # Draw a label with a name below the face_recognition
            #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.imshow("test",frame)
        video.write(frame)
        print(time.time()-start)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

import asyncio
import nats
from nats.aio.errors import ErrConnectionClosed, ErrTimeout, ErrNoServers

import nest_asyncio
nest_asyncio.apply()

import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

import argparse
import os
import platform
import sys
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import base64
import face_recognition 
import scipy.misc
import skimage.transform as st
import pickledb
import functools
from kalmanfilter import KalmanFilter

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, apply_classifier, check_file, check_img_size, check_imshow, check_requirements,
                           check_suffix, colorstr, increment_path, non_max_suppression, print_args, scale_coords,
                           strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import load_classifier, select_device, time_sync
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from matplotlib import pyplot as plt
from digits import plate_segmentation
from nanoid import generate
from datetime import datetime
import json
import time
import math
import sys
import logging



count = 1
deviceId = generate(size=24)
lat = "16.65689"
lon = "12.279779"
time_thread1 = np.array([])
time_thread2 = np.array([])
camera1 = []
camera2 = []
camera3 = []
camera4 = []
face_encoding_store = []
did_store = []

center_points_prev_frame = []


#Initialize Kalman Filter
kf = KalmanFilter()

old_stdout = sys.stdout


async def start_client(delay):
    await asyncio.sleep(delay) 
    # nc = await nats.connect(servers=["nats://164.52.213.244:4222"])
    nc = await nats.connect(servers=["nats://216.48.189.5:4222"])
    js = nc.jetstream()
    print('Connected to NATS...')

    # log_file = open("message.log","w")
    
    print('Loading known faces...')
    known_faces = []
    known_names = []

    db = pickledb.load("knowface.db", True)
    list1 = list(db.getall()) 
    # print(list1)
    
    for name in list1:    
        # Next we load every file of faces of known person
        re_image = db.get(name)

        # Deserialization
        print("Decode JSON serialized NumPy array")
        decodedArrays = json.loads(re_image)

        finalNumpyArray = np.asarray(decodedArrays["array"],dtype="uint8")
        
        # Load an image
        # image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        image = finalNumpyArray
        ratio = np.amax(image) / 256        
        image = (image / ratio).astype('uint8')

        # Get 128-dimension face encoding
        # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
        encoding = face_recognition.face_encodings(image)[0]

        # Append encodings and name
        known_faces.append(encoding)
        known_names.append(name)
    
    async def subscribe_handler(msg):
        # await asyncio.sleep(0)
        global count , time_thread1,time_thread2
        await asyncio.sleep(0.000000001)
        subject = msg.subject 
        reply = msg.reply
        # For C++ code 
        # data =(msg.data)
        # arr = np.ndarray(
        #     (1024,
        #     1024),
        #     buffer=data,
        #     dtype=np.uint8)
        # resized = cv2.resize(arr, (1024, 1024))
        # data = resized
        # for C++
        

        # For Python code 
        data = BytesIO(msg.data)
        data = np.load(data, allow_pickle=True)
        # python 
        im = Image.fromarray(data)
        im.save("./output/output.jpeg")
        await msg.respond(b'ok')
        await msg.ack()




        async def run2(weights=ROOT / 'activity.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=256,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=True,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=True,  # use OpenCV DNN for ONNX inference
        deep_sort_model=True, # deep
        config_deepsort=True,
        ):
            source = str(source)
            global center_points_prev_frame , camera1 , face_encoding_store, did_store
            
            # Directories
            save_dir = Path('./general/'+str(count)+'.jpg')
            save_img = not nosave and not source.endswith('.txt')  # save inference images

            # activity tracking and detection
            class Activity():
                def __init__(self):
                    self.deviceId = deviceId
                    self.lat = lat
                    self.lon = lon
                    # self.timestamp = timestamp
                    self.time = time

            activity = Activity()

            detectList = ["0","0"]
            trackID =[]
            audience =[]
            peopleMotion = []
            direction = []

            
            
            # Initialize
            device = select_device(device)
            half &= device.type != 'cpu'  # half precision only supported on CUDA

            

            # Load model
            w = str(weights[0] if isinstance(weights, list) else weights)
            classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
            check_suffix(w, suffixes)  # check weights have acceptable suffix
            pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
            stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
            if pt:
                model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
                stride = int(model.stride.max())  # model stride
                names = model.module.names if hasattr(model, 'module') else model.names  # get class names
                if half:
                    model.half()  # to FP16
                if classify:  # second-stage classifier
                    modelc = load_classifier(name='resnet50', n=2)  # initialize
                    modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
            elif onnx:
                if dnn:
                    check_requirements(('opencv-python>=4.5.4',))
                    net = cv2.dnn.readNetFromONNX(w)
                else:
                    check_requirements(('onnx', 'onnxruntime-gpu' if torch.has_cuda else 'onnxruntime'))
                    import onnxruntime
                    session = onnxruntime.InferenceSession(w, None)
            else:  # TensorFlow models
                import tensorflow as tf
                if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
                    def wrap_frozen_graph(gd, inputs, outputs):
                        x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
                        return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                                    tf.nest.map_structure(x.graph.as_graph_element, outputs))

                    graph_def = tf.Graph().as_graph_def()
                    graph_def.ParseFromString(open(w, 'rb').read())
                    frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
                elif saved_model:
                    model = tf.keras.models.load_model(w)
                elif tflite:
                    if "edgetpu" in w:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
                        import tflite_runtime.interpreter as tflri
                        delegate = {'Linux': 'libedgetpu.so.1',  # install libedgetpu https://coral.ai/software/#edgetpu-runtime
                                    'Darwin': 'libedgetpu.1.dylib',
                                    'Windows': 'edgetpu.dll'}[platform.system()]
                        interpreter = tflri.Interpreter(model_path=w, experimental_delegates=[tflri.load_delegate(delegate)])
                    else:
                        interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
                    interpreter.allocate_tensors()  # allocate
                    input_details = interpreter.get_input_details()  # inputs
                    output_details = interpreter.get_output_details()  # outputs
                    int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
            imgsz = check_img_size(imgsz, s=stride)  # check image size

            # Dataloader
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
            bs = 1  # batch_size
            vid_path, vid_writer = [None] * bs, [None] * bs

            # Run inference
            if pt and device.type != 'cpu':
                model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
            dt, seen = [0.0, 0.0, 0.0], 0
            for path, img, im0s, vid_cap, s in dataset:
                t1 = time_sync()
                if onnx:
                    img = img.astype('float32')
                else:
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255  # 0 - 255 to 0.0 - 1.0
                if len(img.shape) == 3:
                    img = img[None]  # expand for batch dim
                t2 = time_sync()
                dt[0] += t2 - t1

                # Inference
                if pt:
                    visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                    pred = model(img, augment=augment, visualize=visualize)[0]
                elif onnx:
                    if dnn:
                        net.setInput(img)
                        pred = torch.tensor(net.forward())
                    else:
                        pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
                else:  # tensorflow model (tflite, pb, saved_model)
                    imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
                    if pb:
                        pred = frozen_func(x=tf.constant(imn)).numpy()
                    elif saved_model:
                        pred = model(imn, training=False).numpy()
                    elif tflite:
                        if int8:
                            scale, zero_point = input_details[0]['quantization']
                            imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                        interpreter.set_tensor(input_details[0]['index'], imn)
                        interpreter.invoke()
                        pred = interpreter.get_tensor(output_details[0]['index'])
                        if int8:
                            scale, zero_point = output_details[0]['quantization']
                            pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
                    pred[..., 0] *= imgsz[1]  # x
                    pred[..., 1] *= imgsz[0]  # y
                    pred[..., 2] *= imgsz[1]  # w
                    pred[..., 3] *= imgsz[0]  # h
                    pred = torch.tensor(pred)
                t3 = time_sync()
                dt[1] += t3 - t2

                # NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                dt[2] += time_sync() - t3

                # Second-stage classifier (optional)
                if classify:
                    pred = apply_classifier(pred, modelc, img, im0s)

                

                
                count_person = 0
                count_car = 0
                personDid=[]
                personAlert =[]
                vehicleInvolved=[]
                activityPerson =[]
                VehicleMotion =[] 
                peopleMotion_1 = 0
                TOLERANCE = 0.62
                MODEL = 'cnn' 
                Cnn = load_model('cnn_classifier.h5')
                
                tracking_objects = {}
                track_id = 0
                predicted= [0, 0]
                timestamp = ""
                map_1 = {}

                
                center_points_cur_frame = []

                  
                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # img.jpg
                    txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    imc = im0.copy() if save_crop else im0  # for save_crop
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in reversed(det):

                        if save_img :  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))


                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        cx = float((xywh[0]+xywh[0]+xywh[2])/2)
                        cy = float((xywh[1]+xywh[1]+xywh[3])/2)
                        center_points_cur_frame.append((cx,cy))
                        image_box = cv2.circle(im0, (int(cx*256),int(cy*256)), 10, (255,0,0), -1)
                        cv2.imwrite("./annotation/output"+str(count)+".jpeg",image_box)
                            
                        print(count,"count")
                        if count <= 2:
                            for pt in center_points_cur_frame:
                                for pt2 in center_points_prev_frame:
                                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                                    a = (pt[0] - pt2[0])
                                    b = (pt[1]-pt2[1])
                                    if (a!=0):
                                        z = abs(b/a)
                                        angle = math.degrees(math.atan(z))
                                    else:
                                        angle =90 

                                    if (a<0 and b>0):
                                        anglefinal= 180 - angle
                                        direction.append("far-left")
                                        print("far-left")
                                    
                                    elif (a>0 and  b>0):
                                        anglefinal= angle
                                        direction.append("far-right")
                                        print("Far-right")

                                    elif (a<0 and  b<0):
                                        anglefinal = 180 + angle
                                        direction.append("Near-left")
                                        print("Near-left")

                                    else:
                                        direction.append("Near-right")
                                        print("Near-right")
                                    

                                    if distance < float(20/256):
                                        tracking_objects[track_id] = pt
                                        track_id += 1
                                    else:
                                        predicted = kf.predict(pt[0], pt[1]) 
                                        tracking_objects[track_id] = predicted
                                        track_id += 1
                        else:
                            tracking_objects_copy = tracking_objects.copy()
                            center_points_cur_frame_copy = center_points_cur_frame.copy()

                            for object_id, pt2 in tracking_objects_copy.items():
                                object_exists = False
                                for pt in center_points_cur_frame_copy:
                                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])


                                    if len(tracking_objects.keys()) >= 2: 
                                        a = (pt[0] - pt2[0])
                                        b = (pt[1]-pt2[1])
                                        # print(a , b ,"lower Loop")
                                        if (a!=0):
                                            z = abs(b/a)
                                            angle = math.degrees(math.atan(z))
                                        else:
                                            angle =90 

                                        if (a<0 and b>0):
                                            anglefinal= 180 - angle
                                            direction.append("Far-left")
                                            print("Far-left")
                                        
                                        elif (a>0 and  b>0):
                                            anglefinal= angle
                                            direction.append("Far-Right")
                                            print("Far-right")

                                        elif (a<0 and  b<0):
                                            anglefinal = 180 + angle
                                            direction.append("Near-left")
                                            print("Near-left")

                                        else:
                                            direction.append("Near-right")
                                            print("Near-right")


                                    # Update IDs position
                                    if distance < float(20/256):
                                        tracking_objects[object_id] = pt
                                        object_exists = True
                                        if pt in center_points_cur_frame:
                                            center_points_cur_frame.remove(pt)
                                            continue
                                    else:
                                        predicted = kf.predict(predicted[0], predicted[1])
                                        distance_1 = math.hypot(predicted[0] - pt[0], predicted[1] - pt[1])
                                        if distance_1 < float(25/256): 
                                            tracking_objects[track_id] = predicted
                                            object_exists = True
                                            if pt in center_points_cur_frame:
                                                center_points_cur_frame.remove(pt)
                                                continue

                                # Remove IDs lost
                                if not object_exists:
                                    tracking_objects.pop(object_id)

                            # Add new IDs found
                            for pt in center_points_cur_frame:
                                tracking_objects[track_id] = pt
                                track_id += 1

                                for id in tracking_objects.keys():
                                    if id == 0 :
                                        for pt in center_points_cur_frame:
                                            for pt2 in center_points_prev_frame:
                                                a = (pt[0] - pt2[0])
                                                b = (pt[1]-pt2[1])
                                                # print(a , b ,"lower Loop")
                                                if (a!=0):
                                                    z = abs(b/a)
                                                    angle = math.degrees(math.atan(z))
                                                else:
                                                    angle =90 

                                                if (a<0 and b>0):
                                                    anglefinal= 180 - angle
                                                    direction.append("Far-left")
                                                    print("Far-left")
                                                
                                                elif (a>0 and  b>0):
                                                    anglefinal= angle
                                                    direction.append("Far-Right")
                                                    print("Far-right")

                                                elif (a<0 and  b<0):
                                                    anglefinal = 180 + angle
                                                    direction.append("Near-left")
                                                    print("Near-left")

                                                else:
                                                    direction.append("Near-right")
                                                    print("Near-right")

                    
                    for c in det[:,-1]:
                        print("Detected labels:",names[int(c)])
                        if names[int(c)]=="person":
                            audience.append("0")
                            peopleMotion.append("Static")
                            count_person += 1
                            if count_person>0:
                                np_bytes2 = BytesIO()
                                np.save(np_bytes2, im0, allow_pickle=True)
                                np_bytes2 = np_bytes2.getvalue()
 
                                image = im0 # if im0 does not work, try with im1
                                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                                # print(MODEL, image ,"model ,image")
                                locations = face_recognition.face_locations(image, model=MODEL)
                                # print(locations,"locations")

                                encodings = face_recognition.face_encodings(image, locations)
                                
                                print(f', found {len(encodings)} face(s)\n')
                                
                                for face_encoding ,face_location in zip(encodings, locations):
                                    results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
                                    results_store = face_recognition.compare_faces(face_encoding_store, face_encoding, TOLERANCE)

                                    # print(results, "result")
                                    # print(results_store, "result store")

                                    if True in results_store:
                                        did = did_store[results_store.index(True)]
                                        personDid.append(did)
                                    elif True in results:
                                        match = known_names[results.index(True)]
                                        print("Match found: ", match)
                                        did = generate(size=24)
                                        alertLevel = 0
                                        personDid.append(did)
                                        personAlert.append(alertLevel)
                                    else:
                                        did = generate(size=24)
                                        alertLevel = 1
                                        personDid.append(did)
                                        personAlert.append(alertLevel)

                                    face_encoding_store.append(face_encoding)
                                    did_store.append(did)
                                # print(face_encoding_store,"face encod stor")
                                # print(did_store,"did_stor")

                                   
                        elif names[int(c)]=="vehicle":
                            audience.append("1")
                            VehicleMotion.append("Static")
                            count_car +=1
                            if count_car>0:
                                digits = plate_segmentation("output/output.jpeg")    
                                lp_array = []
                                
                                for d in digits:

                                    d = np.reshape(d, (1,28,28,1))
                                    out = Cnn.predict(d)
                                    # Get max pre arg
                                    p = []
                                    precision = 0
                                    for i in range(len(out)):
                                        z = np.zeros(36)
                                        z[np.argmax(out[i])] = 1.
                                        precision = max(out[i])
                                        p.append(z)
                                    prediction = np.array(p)

                                    # Inverse one hot encoding
                                    alphabets = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
                                    classes = []
                                    for a in alphabets:
                                        classes.append([a])
                                    ohe = OneHotEncoder(handle_unknown='ignore', categories="auto")
                                    ohe.fit(classes)
                                    pred = ohe.inverse_transform(prediction)
                                    
                                    

                                    if precision > 0.6:
                                        print('Prediction : ' + str(pred[0][0]) + ' , Precision : ' + str(precision))
                                        lp_array.append(str(pred[0][0]))
                                        
                                lp_array = ''.join(map(str, lp_array))
                                License_plate = {"class":"License Plate","chars":lp_array}
                                vehicleInvolved.append(License_plate)
                                
                        elif names[int(c)]=="person_in":
                            count_person += 1
                            peopleMotion.append("Person_in")
                            if count_person>0:
                                np_bytes2 = BytesIO()
                                np.save(np_bytes2, im0, allow_pickle=True)
                                np_bytes2 = np_bytes2.getvalue()
 
                                image = im0 # if im0 does not work, try with im1
                                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                                
                                locations = face_recognition.face_locations(image, model=MODEL)
                    

                                encodings = face_recognition.face_encodings(image, locations)
                                
                                print(f', found {len(encodings)} face(s)\n')
                                
                                for face_encoding ,face_location in zip(encodings, locations):
                                    results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
                                    results_store = face_recognition.compare_faces(face_encoding_store, face_encoding, TOLERANCE)


                                    if True in results_store:
                                        did = did_store[results_store.index(True)]
                                        personDid.append(did)
                                    elif True in results:
                                        match = known_names[results.index(True)]
                                        print("Match found: ", match)
                                        did = generate(size=24)
                                        alertLevel = 0
                                        personDid.append(did)
                                        personAlert.append(alertLevel)
                                    else:
                                        did = generate(size=24)
                                        alertLevel = 1
                                        personDid.append(did)
                                        personAlert.append(alertLevel)

                                    face_encoding_store.append(face_encoding)
                                    did_store.append(did)
                                # print(face_encoding_store,"face encod stor")
                                # print(did_store,"did_stor")
                            
        
                                                    
                        elif names[int(c)]=="person_out":
                            count_person += 1
                            peopleMotion.append("Person_out")
                            if count_person>0:
                                np_bytes2 = BytesIO()
                                np.save(np_bytes2, im0, allow_pickle=True)
                                np_bytes2 = np_bytes2.getvalue()
 
                                image = im0 # if im0 does not work, try with im1
                                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                                
                                locations = face_recognition.face_locations(image, model=MODEL)

                                encodings = face_recognition.face_encodings(image, locations)
                                
                                print(f', found {len(encodings)} face(s)\n')
                                
                                for face_encoding ,face_location in zip(encodings, locations):
                                    results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
                                    results_store = face_recognition.compare_faces(face_encoding_store, face_encoding, TOLERANCE)


                                    if True in results_store:
                                        did = did_store[results_store.index(True)]
                                        personDid.append(did)
                                    elif True in results:
                                        match = known_names[results.index(True)]
                                        print("Match found: ", match)
                                        did = generate(size=24)
                                        alertLevel = 0
                                        personDid.append(did)
                                        personAlert.append(alertLevel)
                                    else:
                                        did = generate(size=24)
                                        alertLevel = 1
                                        personDid.append(did)
                                        personAlert.append(alertLevel)

                                    face_encoding_store.append(face_encoding)
                                    did_store.append(did)
                            
                            
                        elif names[int(c)]=="vehicle_in":
                            VehicleMotion.append("vehicle_in")
                            count_car +=1
                            if count_car>0:
                                digits = plate_segmentation("output/output.jpeg")    
                                lp_array = []
                                
                                for d in digits:

                                    d = np.reshape(d, (1,28,28,1))
                                    out = Cnn.predict(d)
                                    # Get max pre arg
                                    p = []
                                    precision = 0
                                    for i in range(len(out)):
                                        z = np.zeros(36)
                                        z[np.argmax(out[i])] = 1.
                                        precision = max(out[i])
                                        p.append(z)
                                    prediction = np.array(p)

                                    # Inverse one hot encoding
                                    alphabets = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
                                    classes = []
                                    for a in alphabets:
                                        classes.append([a])
                                    ohe = OneHotEncoder(handle_unknown='ignore', categories="auto")
                                    ohe.fit(classes)
                                    pred = ohe.inverse_transform(prediction)
                                    
                                    

                                    if precision > 0.6:
                                        print('Prediction : ' + str(pred[0][0]) + ' , Precision : ' + str(precision))
                                        lp_array.append(str(pred[0][0]))
                                        
                                lp_array = ''.join(map(str, lp_array))
                                License_plate = {"class":"License Plate","chars":lp_array}
                                vehicleInvolved.append(License_plate)
                            
                        elif names[int(c)]=="vehicle_out":
                            VehicleMotion.append("vehicle_out")
                            count_car +=1
                            if count_car>0:
                                digits = plate_segmentation("output/output.jpeg")    
                                lp_array = []
                                
                                for d in digits:

                                    d = np.reshape(d, (1,28,28,1))
                                    out = Cnn.predict(d)
                                    # Get max pre arg
                                    p = []
                                    precision = 0
                                    for i in range(len(out)):
                                        z = np.zeros(36)
                                        z[np.argmax(out[i])] = 1.
                                        precision = max(out[i])
                                        p.append(z)
                                    prediction = np.array(p)

                                    # Inverse one hot encoding
                                    alphabets = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
                                    classes = []
                                    for a in alphabets:
                                        classes.append([a])
                                    ohe = OneHotEncoder(handle_unknown='ignore', categories="auto")
                                    ohe.fit(classes)
                                    pred = ohe.inverse_transform(prediction)
                                    
                                    

                                    if precision > 0.6:
                                        print('Prediction : ' + str(pred[0][0]) + ' , Precision : ' + str(precision))
                                        lp_array.append(str(pred[0][0]))
                                        
                                lp_array = ''.join(map(str, lp_array))
                                License_plate = {"class":"License Plate","chars":lp_array}
                                vehicleInvolved.append(License_plate)
                            
                        
                                
                        
                    setattr(activity, "peopleInvolved", personDid)
                    setattr(activity, "peopleAlert", personAlert)
                    setattr(activity, "vehicleInvolved", vehicleInvolved)
                    setattr(activity, "peopleInvolved", personDid)
                    setattr(activity, "peopleAlert", personAlert)
                    setattr(activity, "peopleMotion",peopleMotion)
                    setattr(activity, "VehicleMotion",VehicleMotion)
                    setattr(activity,"audience",audience)
                    setattr(activity,"direction",direction)

                    
                    trackID =list(tracking_objects.keys())
                    setattr(activity, "trackID",trackID)
                    center_points_prev_frame = center_points_cur_frame.copy()
                    lenght_tracking = len(tracking_objects)
                                                
                        
                    
                    timestamp = str(datetime.now())
                    setattr(activity,"timestamp",timestamp)
                    x = count_person
                    y = count_car
                    detectList = [str(x), str(y)] 
                    setattr(activity,"detectList",detectList)
                    map_1[timestamp]= [trackID ,deviceId ,[str(activity.lat),str(activity.lon)],str(activity.peopleInvolved)]
                    camera1.append(map_1)
                    
                
                    
                    jsonActivity = {
                        "deviceId":str(activity.deviceId),
                        "geo":[
                            {"Lat":str(activity.lat), "Lon":str(activity.lon)}
                                ],
                        "timestamp":activity.timestamp,
                        "trackID":activity.trackID,
                        "audience":activity.audience,
                        "peopleInvoled":activity.peopleInvolved,
                        "Direction":activity.direction,
                        "vehicleInvolved":activity.vehicleInvolved,
                        "peopleAlert":activity.peopleAlert,
                        "peopleMotion":activity.peopleMotion,
                        "VehicleMotion":activity.VehicleMotion,
                        }

                    print(jsonActivity)
                    json_encoded = json.dumps(jsonActivity)
                    json_encoded = json_encoded.encode()
                    
                    subjectactivity = "model.activity"
                    await nc.publish(subjectactivity, json_encoded)
                    print("Activity is getting published")

                    LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

                
                    
        srcpath = 'output/output.jpeg'

        async def parse_opt():
            parser = argparse.ArgumentParser()
            parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'activity.pt', help='model path(s)')
            parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
            parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
            parser.add_argument('--source', type=str, default= srcpath, help='file/dir/URL/glob, 0 for webcam')
            parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
            parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
            parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
            parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
            parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
            parser.add_argument('--view-img', action='store_true', help='show results')
            parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
            parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
            parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
            parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
            parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
            parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
            parser.add_argument('--augment', action='store_true', help='augmented inference')
            parser.add_argument('--visualize', action='store_true', help='visualize features')
            parser.add_argument('--update', action='store_true', help='update all models')
            parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
            parser.add_argument('--name', default='exp', help='save results to project/name')
            parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
            parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
            parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
            parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
            parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
            parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
            opt = parser.parse_args()
            opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
            print_args(FILE.stem, opt)
            return opt


        async def main(opt):
            check_requirements(exclude=('tensorboard', 'thop'))
            await run2(**vars(opt))

        opt = await parse_opt()
        await main(opt)


        print("Receiving frames ", subject, count, data.shape)
        count +=1
        
    
    await js.subscribe("device.stream16.frame", "workers", cb=subscribe_handler, stream="sample-stream215")
    # await js.subscribe("device.*.frame", "workers", cb=subscribe_handler , stream="device-stream")
    print("Subscribing to subject...")



async def run():
    
    task1 = asyncio.create_task(start_client(0))
    await task1
 

if __name__ == '__main__':
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
    loop.run_forever()
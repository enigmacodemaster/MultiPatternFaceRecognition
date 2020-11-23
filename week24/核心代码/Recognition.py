# coding:utf-8
from FaceRecog import Recognition
from ReaFace import RealFace

import dlib
import numpy as np
from copy import deepcopy
import cv2
import os


class FaceRecognition(object):
    def __init__(self):
        super(FaceRecognition, self).__init__()
        self.detector = dlib.get_frontal_face_detector()
        self.img_size = 256
        self.face_sproofing = RealFace()
        self.predictor = dlib.shape_predictor(r'./shape_predictor_68_face_landmarks.dat')
        # self.recognition = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
        self.recognition = Recognition()

    def point_draw(self, img, sp, title, save):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for i in range(68):
            cv2.putText(img, str(i), (sp.part(i).x, sp.part(i).y), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 255), 1,
                        cv2.LINE_AA)
            # cv2.drawKeypoints(img, (sp.part(i).x, sp.part(i).y),img, [0, 0, 255])
        if save:
            #filename = title+str(np.random.randint(100))+'.jpg'
            filename = title+'.jpg'
            cv2.imwrite(filename, img)
        os.system("open %s"%(filename)) 
        #cv2.imshow(title, img)
        #cv2.waitKey(0)
        #cv2.destroyWindow(title)

    def show_origin(self, img):
        cv2.imshow('origin', img)
        cv2.waitKey(0)
        cv2.destroyWindow('origin')

    def getfacefeature(self, img):
        import pdb
        pdb.set_trace()
        image = dlib.load_rgb_image(img)
        ## 人脸对齐、切图
        # 人脸检测
        dets = self.detector(image, 1)
        if len(dets) == 1:
            # faces = dlib.full_object_detections()
            # 关键点提取
            shape = self.predictor(image, dets[0])
            print("Computing descriptor on aligned image ..")
            #人脸对齐 face alignment
            images = dlib.get_face_chip(image, shape, size=self.img_size)

            self.point_draw(image, shape, 'before_image_warping', save=True)
            shapeimage = np.array(images).astype(np.uint8)
            dets = self.detector(shapeimage, 1)
            if len(dets) == 1:
                point68 = self.predictor(shapeimage, dets[0])
                self.point_draw(shapeimage, point68, 'after_image_warping', save=True)

            #计算对齐后人脸的128维特征向量
            face_descriptor_from_prealigned_image = self.recognition.compute_face_descriptor(images)
        return face_descriptor_from_prealigned_image

    def compare(self):
        import pdb
        pdb.set_trace()
        vec1 = np.array(self.getfacefeature(self.img_1))
        vec2 = np.array(self.getfacefeature(self.img_2))
        vec3 = np.array(self.getfacefeature(self.img_3))

        import pdb
        pdb.set_trace()
        same_people = np.sqrt(np.sum((vec2-vec3)*(vec2-vec3)))
        not_same_people12 = np.sqrt(np.sum((vec1-vec2)*(vec1-vec2)))
        not_same_people13 = np.sqrt(np.sum((vec1-vec3)*(vec1-vec3)))
        print('distance between different people12:{:.3f}, different people13:{:.3f}, same people:{:.3f}'.\
              format(not_same_people12, not_same_people13, same_people))

#detection_recognition = FaceRecognitionExample(img_1, img_2, img_3)
#detection_recognition.compare()


if __name__=="__main__":
    # 初始化人脸检测模型
    detector = dlib.get_frontal_face_detector()
    ## 填空 初始化活体检测模型
    face_spoofing = RealFace()
    # 初始化关键点检测模型
    predictor = dlib.shape_predictor(r'./shape_predictor_68_face_landmarks.dat')
    # 初始化人脸识别模型
    recognition = Recognition()
    # 从摄像头读取图像, 若摄像头工作不正常，可使用：cv2.VideoCapture("week20_video3.mov"),从视频中读取图像
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("./week20_video3.mov")
    while 1:
        # 初始化人脸相似度为-1
        # similarity=-1
        # 读取图片
        ret, frame_src = cap.read()
        # 将图片缩小为原来大小的1/3
        x, y = frame_src.shape[0:2]
        frame = cv2.resize(frame_src, (int(y / 3), int(x / 3)))
        face_align = frame
        # 使用检测模型对图片进行人脸检测
        dets = detector(frame, 1)
        #import pdb
        #pdb.set_trace()
        # traverse detect results
        for det in dets:
            # 对检测到的人脸提取人脸关键点
            shape=predictor(frame, det)
            #print("x=%s,y=%s,w=%s,h=%s"%(det.left(),det.top(),det.width(),det.height()))
            # 在图片上绘制出检测模型得到的矩形框,框为绿色
            frame=cv2.rectangle(frame,(det.left(),det.top()),(det.right(),det.bottom()),(0,255,0),2)
            #import pdb
            #pdb.set_trace()
            # 人脸对齐
            face_align=dlib.get_face_chip(frame, shape, 150,0.1)
            ## 活体检测
            is_real = 'real'
            if not face_spoofing.classify(face_align):
                print("fake !!! \n ")     
                # 框为红色
                is_real = 'fake'
                frame=cv2.rectangle(frame,(det.left(),det.top()),(det.right(),det.bottom()),(0,0,255),2)
            # 检测人脸
            is_same = "SAME"
            recog_result = recognition.detect(face_align, base_img, threshhold = 0.73)
            if not recog_result:
            	is_same = "DIFFERENT"
            # 计算人脸相似度
            # similarity=1 - np.linalg.norm(np.array(face_feature)-np.array(face_feature_zmm))
            # 将关键点绘制到人脸上，
            for i in range(68):
                cv2.putText(frame, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 255, 255), 1,cv2.LINE_AA)
                cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255))
        #print(dets.rectangles)
        # 为了显示出相似度，我们将相似度写到图片上，
        cv2.putText(frame,"%s - %s"%(is_same, is_real),(100,200),cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 255, 255), 1,cv2.LINE_AA)
        # 显示图片，图片上有矩形框，关键点，以及相似度
        cv2.imshow("capture", cv2.resize(frame,(y // 3,x // 3)))
        #cv2.imshow("face_align",face_align)
        if cv2.waitKey(100) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

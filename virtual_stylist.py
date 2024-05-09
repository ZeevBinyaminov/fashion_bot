# !brew install cmake
# !brew install boost
# !brew install boost-python
# !brew install dlib
# !pip3 install numpy
# !pip3 install scipy
# !pip3 install scikit-image
# !pip3 install dlib


# !brew install boost-python3
# !brew install --cask boost-note


import math
import os
import cv2
import dlib
import imutils
import numpy as np
import pandas as pd
from imutils import face_utils
from imutils.face_utils import FaceAligner
from scipy.spatial import distance as dist
from werkzeug.utils import secure_filename


# pip install opencv-python

# pip install dlib


ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'JPG'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# In[7]:


def get_angle_abc(A, B, C):
    a = np.radians(np.array(A))
    b = np.radians(np.array(B))
    c = np.radians(np.array(C))
    avec = a - b
    cvec = c - b
    lat = b[0]
    avec[1] *= math.cos(lat)
    cvec[1] *= math.cos(lat)
    return np.degrees(math.acos(np.dot(avec, cvec) / (np.linalg.norm(avec) * np.linalg.norm(cvec))))


# In[8]:


def typing_result_text(R, D, G, N):
    typing_key_zero = (int(bool(R)) + int(bool(D)) +
                       int(bool(G)) + int(bool(N)))
    if typing_key_zero <= 1:
        typing_key = int(bool(R)), int(bool(D)), int(bool(G)), int(bool(N))
    dict_typing_text = {(1, 0, 0, 0): 'romantic',
                        (0, 1, 0, 0): 'dramatic',
                        (0, 0, 1, 0): 'gamin',
                        (0, 0, 0, 1): 'natural',
                        (0, 0, 0, 0): 'manual definition'}
    if typing_key in dict_typing_text:
        return dict_typing_text[typing_key]
    if typing_key_zero == 2:
        typing_key = (int(bool(R)), int(bool(D)), int(bool(G)), int(bool(N)),
                      int(R <= D), int(D < R),
                      int(R <= G), int(G < R),
                      int(R <= N), int(N < R),
                      int(D <= G), int(G < D),
                      int(D <= N), int(N < D),
                      int(G <= N), int(N < G)
                      )
    dict_typing_text = {
        (1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0): 'romantic dramatic',
        (1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0): 'dramatic romantic',
        (1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1): 'romantic gamin',
        (1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1): 'gamin romantic',
        (1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0): 'romantic natural',
        (1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0): 'natural romantic',
        (0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1): 'dramatic gamin',
        (0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1): 'gamin dramatic',
        (0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0): 'dramatic natural',
        (0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0): 'natural dramatic',
        (0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0): 'gamin natural',
        (0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1): 'natural gamin',
    }
    if typing_key in dict_typing_text:
        return dict_typing_text[typing_key]
    if typing_key_zero > 2:
        if (R == D) & (R > G) & (R > N):
            return 'romantic dramatic'
        if (R == G) & (R > D) & (R > N):
            return 'romantic gamin'
        if (R == N) & (R > D) & (R > G):
            return 'romantic natural'
        if (D == G) & (D > R) & (D > N):
            return 'dramatic gamin'
        if (D == N) & (D > R) & (D > G):
            return 'dramatic natural'
        if (G == N) & (G > R) & (G > D):
            return 'gamin natural'
        if (R > D) & (R > G) & (R > N):
            return 'romantic classic'
        if (D > R) & (D > G) & (D > N):
            return 'dramatic classic'
        if (G > R) & (G > D) & (G > N):
            return 'gamin classic'
        if (N > R) & (N > D) & (N > G):
            return 'natural classic'
    return 'manual definition'


# In[9]:


def add_df_dist_68_to_excel(shape, image, excel):
    matrix_dist = [[dist.euclidean(shape[i], shape[j])
                    for j in range(68)] for i in range(68)]
    name_index = [0] * 68 * 68
    vector_dist = [0] * 68 * 68
    iterator_dist = -1
    for i in range(68):
        for j in range(68):
            iterator_dist += 1
            vector_dist[iterator_dist] = matrix_dist[i][j]
            name_index[iterator_dist] = ('{}-{}'.format(i, j))
        df0 = pd.read_excel(excel, index_col=0)
        df0[(image)] = pd.Series(vector_dist, index=name_index)
        df0.to_excel(excel)
    return


# In[10]:


def faceAligned(filename, shape_predictor):
    # initialize dlib's face detector (HOG-based) and then create # the facial landmark predictor and the face aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    fa = FaceAligner(predictor, desiredFaceWidth=800)
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(filename)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # show the original input image and detect faces in the grayscale
    # image
    rects = detector(gray, 2)
    # loop over the face detections
    for rect in rects:
        faceAligned = fa.align(image, gray, rect)
        cv2.imwrite('test/'+filename, faceAligned)
    return


# ВЫСОТА HEIGHT
# ШИРИНА ПЛЕЧ SHOULDER WIDTH
# ОКРУЖНОСТЬ ТАЛИИ  WAIST CIRCUMFERENCE
# ОКРУЖНОСТЬ ГРУДИ CHEST CIRCUMFERENCE
# ОБХВАТ БЁДЕР HIP CIRCUMFERENCE

def figure_correction(height, shoulder_width, neck_length,
                      bust_length, waist_length, hip_length,
                      underhip_length, underbust_length, leg_length, body_length):
    # correction bust
    if bust_length <= underbust_length+10:
        correction_bust = 'Correction: small breast'
    elif underbust_length+10 < bust_length < (height/2+2.5):
        correction_bust = 'breast correction is not required'
    elif (height/2+2.5) <= bust_length:
        correction_bust = 'Correction: large breast'

    # correction waist
    if waist_length < (height - 100):
        correction_waist = 'waist correction is not required'
    elif waist_length >= (height - 100):
        correction_waist = 'Correction: waist'
    # correction hip
    if hip_length < (waist_length/0.7):
        correction_hip = 'hip correction is not required'
    elif hip_length >= (waist_length/0.7):
        correction_hip = 'Correction: hip'

    # correction leg
    if leg_length - body_length > 15:
        correction_leg = 'hip correction is not required'
    elif leg_length - body_length <= 15:
        correction_leg = 'Correction: short leg'

    # correction shoulder
    if shoulder_width < 16:
        correction_shoulder = 'shoulder correction is not required'
    elif shoulder_width >= 16:
        correction_shoulder = 'Correction: broad shoulders'
    correction = (correction_bust, correction_waist,
                  correction_hip, correction_leg, correction_shoulder)
    return correction


# In[12]:

# ВЫСОТА
# ШИРИНА ПЛЕЧ
# ОКРУЖНОСТЬ ТАЛИИ
# ОКРУЖНОСТЬ ГРУДИ
# ОБХВАТ БЁДЕР

def test_typing(filename, shape_predictor, excel,
                height, shoulder_width, neck_length,
                bust_length, waist_length, hip_length,
                underhip_length, underbust_length, leg_length, body_length):
    # initialize dlib's face detector (HOG-based) and then create # the facial landmark predictor and the face aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    fa = FaceAligner(predictor, desiredFaceWidth=800)
    image = cv2.imread(filename)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    Romantik, Dramatik, Gamin, Natural = 0, 0, 0, 0
    Bal_Natural_Glaza = (dist.euclidean(shape[39], shape[42]) /
                         dist.euclidean(shape[1], shape[16]))
    Bal_Dramatik_Glaza = (dist.euclidean(shape[37], shape[41]) /
                          dist.euclidean(shape[36], shape[39]))  # dist_38_42 / dist_37_40

    Bal_Romantik_Glaza = ((dist.euclidean(shape[37], shape[41]) +
                           dist.euclidean(shape[38], shape[40]) +
                           dist.euclidean(shape[36], shape[39])) /
                          dist.euclidean(shape[0], shape[16]))  # (dist_38_42 + dist_39_41 + dist_37_40) / dist_1_17
    Bal_Gamin_Glaza = (dist.euclidean(shape[37], shape[41]) + dist.euclidean(shape[38], shape[40])) / \
        (dist.euclidean(shape[0], shape[8]) +
         dist.euclidean(shape[16], shape[8]))  # (dist_38_42 + dist_39_41) / (dist_1_9 + dist_17_9)

    if height > 173:
        Dramatik += 1

    if 169 <= height < 173:
        Natural += 1

    if 165 < height < 169:
        Romantik += 1

    if height <= 165:
        Gamin += 1

    if Bal_Natural_Glaza > 0.261:
        Natural += 1

    if Bal_Dramatik_Glaza < 0.35:
        Dramatik += 1

    if Bal_Romantik_Glaza <= 0.3 and Bal_Dramatik_Glaza >= 0.35:
        Romantik += 1

    if Bal_Gamin_Glaza >= 0.082 and Bal_Dramatik_Glaza >= 0.35:
        Gamin += 1

    Bal_Rot_1 = (dist.euclidean(shape[50], shape[58]) /
                 dist.euclidean(shape[48], shape[54]))  # (51-59)/(49-55)
    Bal_Rot_2 = (dist.euclidean(shape[53], shape[55]) /
                 dist.euclidean(shape[51], shape[59]))  # (54-56)/(51-59)
    if Bal_Rot_1 <= 0.39:
        Dramatik += 1

    if Bal_Rot_1 > 0.432 and Bal_Rot_2 > 69:
        Romantik += 1

    if 0.39 < Bal_Rot_1 < 0.432:
        Gamin += 1

    if Bal_Rot_1 > 0.432 and Bal_Rot_2 < 69:
        Natural += 1

    Bal_Nos_1 = (dist.euclidean(
        shape[31], shape[35]) / dist.euclidean(shape[0], shape[16]))
    # dist_32_36 / dist_1_17
    Bal_Nos_2 = (dist.euclidean(
        shape[27], shape[33]) / dist.euclidean(shape[0], shape[8]))
    # dist_28_34 / dist_1_9
    if Bal_Nos_1 >= 0.172:
        Dramatik += 1

    if 0.149 < Bal_Nos_1 < 0.172:
        Natural += 1

    if Bal_Nos_1 <= 0.149 and Bal_Nos_2 > 0.35:
        Romantik += 1

    if Bal_Nos_1 <= 0.149 and Bal_Nos_2 <= 0.35:
        Gamin += 1

    # dist_18_22 / dist_1_17
    Bal_Brovi = (dist.euclidean(
        shape[17], shape[21]) / dist.euclidean(shape[0], shape[16]))
    # dist_18_20 / dist_22_20
    Bal_Brovi_2 = (dist.euclidean(
        shape[17], shape[19]) / dist.euclidean(shape[21], shape[19]))
    if 0.295 <= Bal_Brovi <= 0.347 and Bal_Brovi_2 <= 0.945:
        Dramatik += 1

    if 0.295 <= Bal_Brovi <= 0.347 and Bal_Brovi_2 > 0.945:
        Natural += 1

    if Bal_Brovi <= 0.295:
        Romantik += 1

    if Bal_Brovi >= 0.347:
        Gamin += 1

    # dist_5_13 / dist_1_9 Bal_Natural_lico = True
    Bal_Lico_1 = (dist.euclidean(
        shape[4], shape[12]) / dist.euclidean(shape[0], shape[8]))
    if Bal_Lico_1 < 0.88:
        Dramatik += 1
        Bal_Natural_lico = False

    if 1.02 > Bal_Lico_1 > .96:
        Gamin += 1
        Bal_Natural_lico = False

    if Bal_Lico_1 > 1.02:
        Romantik += 1
        Bal_Natural_lico = False

    if Bal_Natural_lico:
        Natural += 1

    nomer_iteration = 0
    for (x, y) in shape:
        cv2.circle(image, (x, y), 3, (0, 255, 255), -1)
        cv2.putText(image, "{}".format(nomer_iteration), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)
        nomer_iteration += 1

    # draw the total number of blinks on the frame along with
    # the computed eye aspect ratio for the frame
    text_img = typing_result_text(Romantik, Dramatik, Gamin, Natural)
    corrections = figure_correction(height, shoulder_width, neck_length, bust_length, waist_length, hip_length,
                                    underhip_length, underbust_length, leg_length, body_length)

    return text_img

    # cv2.putText(image, "{}".format(text_img), (10, 50),
    #             cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    # nomer_iteration = 0
    # for correction in corrections:
    #     cv2.putText(image, "{}".format(correction),
    #                 (10, 790-30*nomer_iteration), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #     nomer_iteration += 1

    # result_result = filename + "result.png"
    # cv2.imwrite(result_result, image)
    # return filename + "result.png"


# height = int(request.form.get('input_height'))  # Рост
# shoulder_width = int(request.form.get('input_shoulder_width'))
# neck_length = int(request.form.get(
#     'input_neck_length'))  # Окружностьшеи
# bust_length = int(request.form.get('input_bust_length'))  # Обхватгруди
# waist_length = int(request.form.get(
#     'input_waist_length'))  # Обхватталии
# hip_length = int(request.form.get('input_hip_length'))  # Обхватбедер
# underhip_length = int(request.form.get(
#     'input_underhip_length'))  # Объембедерподягодицами
# underbust_length = int(request.form.get(
#     'input_underbust_length'))  # Обхватподгрудью
# leg_length = int(request.form.get('input_leg_length')
#                  )  # Длинаноготпахадоступни
# # Длинателаотсолнечногосплетениядопаха
# body_length = int(request.form.get('input_body_length'))

# filename = ''

# shape_predictor = "shape_predictor_68_face_landmarks.dat"
# excel = "tabl_dist.xlsx"
# faceAligned(filename, shape_predictor)
# result = test_typing(filename, shape_predictor, excel, height, shoulder_width, neck_length,
#                      bust_length, waist_length, hip_length, underhip_length,
#                      underbust_length, leg_length, body_length)
# return redirect(url_for('uploaded_file', filename=result))

import os
from tqdm import tqdm
import cv2
import insightface
import threading
import roop.globals
from roop.analyser import get_face_single, get_face_many

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()

def get_face_swapper():
    global FACE_SWAPPER
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.providers)
    return FACE_SWAPPER

def swap_face_in_frame(source_face, target_face, frame):
    if target_face:
        return get_face_swapper().get(frame, target_face, source_face, paste_back=True)
    return frame

def process_faces(source_face, target_frame):
    if roop.globals.all_faces:
        many_faces = get_face_many(target_frame)
        if many_faces:
            for face in many_faces:
                target_frame = swap_face_in_frame(source_face, face, target_frame)
    else:
        face = get_face_single(target_frame)
        if face:
            target_frame = swap_face_in_frame(source_face, face, target_frame)
    return target_frame

def process_frames(source_img, frame_paths, progress=None):
    source_face = get_face_single(cv2.imread(source_img))
    for frame_path in tqdm(frame_paths, desc="Processing", unit="frame", dynamic_ncols=True):
        frame = cv2.imread(frame_path)
        try:
            result = process_faces(source_face, frame)
            cv2.imwrite(frame_path, result)
        except Exception as exception:
            print(exception)
            pass
        if progress:
            progress.update(1)

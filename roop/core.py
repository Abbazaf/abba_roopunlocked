import os
import sys
import signal
import shutil
import glob
import argparse
import cv2
from pathlib import Path
import multiprocessing as mp
import roop.globals
from roop.swapper import process_video, process_faces
from roop.utils import detect_fps, set_fps, create_video, add_audio, extract_frames, rreplace
import roop.ui as ui

signal.signal(signal.SIGINT, lambda signal_number, frame: quit())
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--face', help='use this face', dest='source_img')
parser.add_argument('-t', '--target', help='replace this face', dest='target_path')
parser.add_argument('-o', '--output', help='save output to this file', dest='output_file')
parser.add_argument('--keep-fps', help='maintain original fps', dest='keep_fps', action='store_true', default=False)
parser.add_argument('--keep-frames', help='keep frames directory', dest='keep_frames', action='store_true', default=False)
parser.add_argument('--all-faces', help='swap all faces in frame', dest='all_faces', action='store_true', default=False)
parser.add_argument('--max-memory', help='maximum amount of RAM in GB to be used', dest='max_memory', type=int)
parser.add_argument('--cpu-cores', help='number of CPU cores to use', dest='cpu_cores', type=int, default=max(mp.cpu_count() / 2, 1))
parser.add_argument('--gpu-threads', help='number of threads to be use for the GPU', dest='gpu_threads', type=int, default=8)
parser.add_argument('--gpu-vendor', help='choice your GPU vendor', dest='gpu_vendor', choices=['apple', 'amd', 'intel', 'nvidia'])

args = parser.parse_known_args()[0]

if 'all_faces' in args:
    roop.globals.all_faces = True

if args.cpu_cores:
    roop.globals.cpu_cores = int(args.cpu_cores)

sep = "/"
if os.name == "nt":
    sep = "\\"

# Optimizing face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def limit_resources():
    gpus = cv2.cuda.getCudaEnabledDeviceCount()
    if gpus > 0:
        cv2.setNumThreads(0)  # Use all CPU threads
        cv2.setUseOptimized(True)  # Enable optimized OpenCV functions
    if args.max_memory:
        memory = args.max_memory * 1024 * 1024 * 1024
        if sys.platform == 'win32':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def pre_check():
    # Your pre-check logic here
    pass

def process_faces_optimized(source_img, frame):
    faces = detect_face(frame)
    for (x, y, w, h) in faces:
        # Process each detected face
        face_img = frame[y:y+h, x:x+w]
        # Your face swapping logic here

def start(preview_callback=None):
    # Your start function implementation here
    pass

def run():
    pre_check()
    limit_resources()
    if args.source_img:
        args.cli_mode = True
        start()
        quit()

    window = ui.init(
        {
            'all_faces': roop.globals.all_faces,
            'keep_fps': args.keep_fps,
            'keep_frames': args.keep_frames
        },
        select_face_handler,
        select_target_handler,
        toggle_all_faces_handler,
        toggle_fps_limit_handler,
        toggle_keep_frames_handler,
        save_file_handler,
        start,
        get_video_frame,
        create_test_preview
    )

    window.mainloop()


if __name__ == '__main__':
    run()

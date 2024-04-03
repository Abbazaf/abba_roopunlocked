#!/usr/bin/env python3

import os
import sys
import signal
import shutil
import glob
import argparse
import cv2
import torch
import tensorflow as tf
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

# Set OMP_NUM_THREADS for better GPU performance
if any(arg.startswith('--gpu-vendor') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'

# GPU thread fix for AMD
if args.gpu_vendor == 'amd':
    roop.globals.gpu_threads = 1

if args.gpu_vendor:
    roop.globals.gpu_vendor = args.gpu_vendor
else:
    roop.globals.providers = ['CPUExecutionProvider']

sep = "/"
if os.name == "nt":
    sep = "\\"


def limit_resources():
    # Prevent TensorFlow memory leak
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Limit maximum RAM usage if specified
    if args.max_memory:
        memory = args.max_memory * 1024 * 1024 * 1024
        if sys.platform == 'win32':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def get_video_frame(video_path, frame_number=1):
    cap = cv2.VideoCapture(video_path)
    amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(amount_of_frames, frame_number-1))
    if not cap.isOpened():
        print("Error opening video file")
        return
    ret, frame = cap.read()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cap.release()


def status(string):
    value = "Status: " + string
    if 'cli_mode' in args:
        print(value)
    else:
        ui.update_status_label(value)


def process_video_multi_cores(source_img, frame_paths):
    n = len(frame_paths) // roop.globals.cpu_cores
    if n > 2:
        processes = []
        for i in range(0, len(frame_paths), n):
            p = POOL.apply_async(process_video, args=(source_img, frame_paths[i:i + n],))
            processes.append(p)
        for p in processes:
            p.get()
        POOL.close()
        POOL.join()


def start(preview_callback=None):
    # Your start function implementation here
    pass


def select_face_handler(path: str):
    args.source_img = path


def select_target_handler(path: str):
    args.target_path = path
    return preview_video(args.target_path)


def toggle_all_faces_handler(value: int):
    roop.globals.all_faces = True if value == 1 else False


def toggle_fps_limit_handler(value: int):
    args.keep_fps = int(value != 1)


def toggle_keep_frames_handler(value: int):
    args.keep_frames = value


def save_file_handler(path: str):
    args.output_file = path


def create_test_preview(frame_number):
    return process_faces(
        get_face_single(cv2.imread(args.source_img)),
        get_video_frame(args.target_path, frame_number)
    )


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

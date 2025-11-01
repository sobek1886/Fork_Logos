"""
Convert data from mp4 videos + txt file for VideoDataset to hdf5video dataset file
"""
from argparse import ArgumentParser
import ctypes
from multiprocessing import Process, Queue, Value, cpu_count
import pandas as pd
from pathlib import Path
import random
from time import sleep
from typing import List

from tqdm import tqdm

from mmengine.fileio import exists, list_from_file

from mmaction.datasets.transforms.hdf5video import Hdf5VideoWriter, extend_path, encode_video


def worker_fn(workqueue, encode_func, done_number, out_queue):
    """
    Worker function for multiptocess conversion.
    Process one video from workqueue with encode_func and return results in out_queue.
    Increments done_number
    """
    while True:
        chunk = workqueue.get()
        if chunk is None:
            # No more work left to do
            break
        file_name, label = chunk
        data, meta = encode_func(file_name)
        out_queue.put((file_name, label, data, meta,))
        with done_number.get_lock():
            done_number.value += 1
        #print(file_name, label)


def write_parallel(file_data, writer, args):
    """
    convert videos from file_data list and write to writer
    """
    num_samples = len(file_data)

    workqueue: Queue = Queue()
    for todo in file_data:
        workqueue.put(todo)

    # We add a token for each worker to warn them that there
    # is no more work to be done
    for _ in range(args.num_workers):
        workqueue.put(None)

    outqueue: Queue = Queue()

    done_number = Value(ctypes.c_uint64, 0)

    def encode_func(video_filename):
        video_full_path = extend_path(writer.video_root_path, video_filename)
        try:
            data, meta = encode_video(
                video_full_path = video_full_path,
                #write_mode=,
                max_resolution = writer.max_resolution,
                min_resolution = writer.min_resolution,
                jpeg_quality = writer.jpeg_quality,
                interpolation = writer.interpolation)
        except Exception as e:
            print(f'\nERROR: failed to encode video {video_full_path}\n{e}')
            if args.ignore_errors:
                return None, None
            else:
                raise
        return data, meta

    # Arguments that have to be passed to the workers
    worker_args = (workqueue, encode_func, done_number, outqueue)

    # Create the workers
    processes = [Process(target=worker_fn, args=worker_args)
                    for _ in range(args.num_workers)]
    # start the workers
    for p in processes:
        p.start()
    # Wait for all the workers to be done

    # # Display progress
    progress = tqdm(total=num_samples)
    previous = 0
    while done_number.value != num_samples or not outqueue.empty():
        val = done_number.value
        diff = val - previous
        if diff > 0:
            progress.update(diff)
        previous = val
        if not outqueue.empty():
            while not outqueue.empty():
                res = outqueue.get()  # (video_filename, label, data, meta)
                if res[-2] is not None:
                    writer.save_data(*res)
        else:
            sleep(0.1)
    progress.close()

    # Wait for all the workers to be done and get their allocations
    for p in processes:
        p.join()

    print('done')
    

def load_logos_to_data_list(input_file):
    df = pd.read_csv(input_file, sep='\t')
    filenames = (df['attachment_id']+'.mp4').to_list()
    gloss_ids = df['gloss'].apply(int).apply(str).to_list()
    starts = df['begin'].apply(int).apply(str).to_list()    
    ends = df['end'].apply(int).apply(str).to_list()
    label = map(lambda x: list(x), zip(gloss_ids, starts, ends))
    data = list(zip(filenames, label))
    return data

def load_data_list(input_file, args, delimiter: str = ' ') -> List[List]:
    """
    Load mmaction2 annotation file
    retrun list of (filename, label)
    label can be int or list of ints
    """
    if args.format == "logos":
        return load_logos_to_data_list(input_file)
    assert args.format in ("", "slovo") 
    
    exists(input_file)
    data_list = []
    fin = list_from_file(input_file)
    for line in fin:
        line_split = line.strip().split(delimiter)
        filename, label = line_split[0], line_split[1:]
        data_list.append((filename, label,))
    return data_list

def run(input_file, output_file, args):
    file_data = load_data_list(input_file, args)
    random.shuffle(file_data)

    video_root_path = Path(args.video_root)
    if not video_root_path.is_absolute():
        video_root_path = Path(input_file).parent / video_root_path
    writer = Hdf5VideoWriter(output_file, video_root_path,
                            max_resolution=args.max_resolution,
                            min_resolution=args.min_resolution,
                            jpeg_quality=args.jpeg_quality)
    write_parallel(file_data, writer, args)
    print(f'dataset saved to {str(output_file)}')

if __name__=="__main__":
    DEFAULT_OUTPUT = None

    # define parameters inline here:

    DEFAULT_INPUT = '/home/jovyan/ovodov/data/slovo_debug/slovo_debug.txt'

    DEFAULT_INPUT = '/home/jovyan/datasets/rsl/slovo/full/full_trimmed/slovo_test.txt'
    DEFAULT_OUTPUT = '/home/jovyan/ovodov/Cifar10Example/data/slovo_test_jpg90_300.hdf5video'

    ######################3

    parser = ArgumentParser()
    parser.add_argument(
        'input_file', type=str, nargs='?', help='Input txt file in VideoDataset format')
    parser.add_argument(
        'output_file', type=str, nargs='?', help='Output hdf5video file (default: input file with extention replaced to .hdf5video)')
    parser.add_argument(
        '--format',
        type=str,
        default="",
        help='Format: slovo(default), "logos"')
    parser.add_argument(
        '--video_root',
        type=str,
        default="",
        help='Video root path (absolute or relative to input_file), default:""')
    parser.add_argument(
        '--max_resolution',
        type=int,
        default=300,
        help='Output max resolution, default:300')
    parser.add_argument(
        '--min_resolution',
        type=int,
        default=None,
        help='Output min resolution, default:None')    
    parser.add_argument(
        '--jpeg_quality',
        type=int,
        default=95,
        help='Output jpeg quality, default:95')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=6,
        help='Num. of process workeras, default:6')
    parser.add_argument(
        '--ignore_errors',
        action='store_true',
        help='Ignore errors decoding video (just exclude from resulting dataset)')
    args = parser.parse_args()

    args.input_file = args.input_file or DEFAULT_INPUT
    args.output_file = args.output_file or DEFAULT_OUTPUT or str(Path(args.input_file).with_suffix('.hdf5video'))

    run(args.input_file, args.output_file, args)


    
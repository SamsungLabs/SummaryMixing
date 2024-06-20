""" SummaryMixing © 2023 by Samsung Electronics is licensed under CC BY-NC 4.0.

Usage: Install SpeechBrain
       Create a folder recipes/Libri-Light/self-supervised-learning/wav2vec2 
       Copy this file under recipes/Libri-Light/self-supervised-learning/wav2vec2 

This is script create speebrain csv files for the Libri-Light dataset
1. download the Libri-Light dataset through the toolkit in the Libri-Light github repo
2. use the vad script from the Libri-Light repo to do the vad
3. use "python make_librilight_csv.py path_to_vad_output path_to_save_csv" to generate the train.csv for the SSL pretraining 


SummaryMixing: https://arxiv.org/abs/2307.07421
SummaryMixing SSL: 

Authors
 * Titouan Parcollet 2023, 2024
 * Shucong Zhang 2023, 2024
 * Rogier van Dalen 2023, 2024
 * Sourav Bhattacharya 2023, 2024
"""


import pathlib
import torchaudio
import tqdm
import multiprocessing
import csv
from pathlib import Path
import sys
import os 

def make_csv_for_each(subpath_1_csv_file_folder, max_length=20.2):
    subpath_1, csv_file_folder = subpath_1_csv_file_folder
    for i, flac_file in enumerate(subpath_1.glob('**/*.flac')):
        flac_file_name = flac_file.stem
        waveform, sample_rate = torchaudio.load(str(flac_file))
        num_frames = waveform.size(1)
        duration_seconds = num_frames / sample_rate
        if duration_seconds > max_length:
            continue
        audio_length_seconds = waveform.shape[1] / sample_rate
        csv_file = f"{csv_file_folder}/{flac_file.parent.stem}.csv"
        with open(csv_file, mode='a', newline='') as csvfile:
            csv_writer = csv.writer(
                csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerow([flac_file_name,audio_length_seconds,str(flac_file)])


def processes_folder(data_path, csv_file_folder):
    os.makedirs(csv_file_folder, exist_ok=True)
    os.makedirs(f"{csv_file_folder}/tmp", exist_ok=True)
    list_dir = pathlib.Path(data_path)
    tasks = []
    for x in list_dir.iterdir():
        tasks.append((x, f"{csv_file_folder}/tmp"))
    with multiprocessing.Pool(processes=128) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(make_csv_for_each, tasks), total=len(tasks)):
            pass

def merge_csv_files(csv_file_folder):
    file_list = [str(x) for x in Path(f"{csv_file_folder}/tmp").glob('*.csv')]
    output_file = f"{csv_file_folder}/train.csv"
    fieldnames = ["ID", "duration", "wav"]  

    with open(output_file, mode='a', newline='', encoding='utf-8') as outfile:
        csv_writer = csv.writer(
                    outfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                )
        csv_writer.writerow(fieldnames)
        for file_path in tqdm.tqdm(file_list):
            with open(file_path, mode='r', encoding='utf-8') as infile:
                reader = csv.reader(infile)
                
                # filter out bad rows
                for row in reader:
                    if len(row) == 3 and os.path.exists(row[-1]):
                        new_row = [row[-1], row[1], row[2]]
                        csv_writer.writerow(new_row)
                    else:
                        print(f"bad row {row}")

    import shutil
    shutil.rmtree(f"{csv_file_folder}/tmp")


if __name__ == "__main__":
    processes_folder(sys.argv[1], sys.argv[2])
    merge_csv_files(sys.argv[2])

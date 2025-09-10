import os
import re
import json
from collections import defaultdict
import subprocess
import time

import numpy as np
import pandas as pd
from tqdm import tqdm



# print(subprocess.check_output('nvidia-smi', shell=True).decode())


def get_free_gpu_IDs_by_mem(memory_threshold=1,exclude=[]):
    nvidia_smi_out = subprocess.check_output('nvidia-smi', shell=True).decode()
    memory_list = re.findall(r'W \|\s*(\d+)MiB / ', nvidia_smi_out)
    available_gpu_IDs = []
    for gpu_ID, memory in enumerate(memory_list):
        if gpu_ID in exclude:
            continue
        if float(memory)/1024 < memory_threshold:
            available_gpu_IDs.append(gpu_ID)
    return available_gpu_IDs
# get_free_gpu_IDs_by_mem(memory_threshold=1,exclude=[])


def get_free_gpu_IDs_by_util(util_threshold=5, runs=20,exclude=[]):
    sum_util_dict = defaultdict(int)
    for i in tqdm(range(runs)):
        nvidia_smi_out = subprocess.check_output('nvidia-smi', shell=True).decode()
        util_list = re.findall(r'MiB \|\s*(\d+)\%\s*Default', nvidia_smi_out)
        for gpu_ID, util in enumerate(util_list):
            sum_util_dict[gpu_ID] += float(util)
    for gpu_ID, _ in enumerate(util_list):
        sum_util_dict[gpu_ID] = sum_util_dict[gpu_ID] / runs
    available_gpu_IDs = []
    for gpu_ID, util in sum_util_dict.items():
        if gpu_ID in exclude:
            continue
        if util < util_threshold:
            available_gpu_IDs.append(gpu_ID)
    return available_gpu_IDs
# get_free_gpu_IDs_by_util()


def print_concole_output(concole_output_dict):
    for idx, concole_output in concole_output_dict.items():
        print(idx)
        output_lines = concole_output[1].decode().split(']')
        output_lines = [l for l in output_lines if not re.search('s/it', l)]
        output_lines = [l for l in output_lines if not re.search('it/s', l)]
        print('len(output_lines)', len(output_lines))
        for l in output_lines:
            print(l)
        
        
#%%
import argparse
import glob
import os
import random
from pathlib import Path
import shutil
import math

import numpy as np

from utils import get_module_logger

def split(source, destination, p_val, p_test):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
        - p_val [float]: proportion of used dataset for validation, value between 0 and 1
        - p_test [float]: proportion of used dataset for test, value between 0 and 1
    """
    train=destination+"/train"
    val=destination+"/val"
    test=destination+"/test"
    
    try:
        # Create target Directory
        os.mkdir(train)
        os.mkdir(val)
        os.mkdir(test)
        print("Directories " , train, val, test ,  " Created ") 
    except FileExistsError:
        print("Directories " , train, val, test  ,  " already exists")

    p_val=float(p_val)
    p_test=float(p_test)
    #take data froç source and shuffle it
    files=os.listdir(source)
    random.shuffle(files)

    n_files=len(files)
    n_val=math.ceil(n_files*p_val)
    n_test=math.ceil(n_files*p_test)
    n_train=n_files-n_val-n_test

    # iterating over all the files in the source directory
    for idx, fname in enumerate(files):
        
        if idx<n_test:
            # copying the files to the destination test
            shutil.copy2(os.path.join(source,fname), test)
        elif idx<n_test+n_val:
            # copying the files to the destination validation
            shutil.copy2(os.path.join(source,fname), val)
        else:
            # copying the files to the destination train
            shutil.copy2(os.path.join(source,fname), train)


if __name__ == "__main__":
    # python create_splits.py --destination data/waymo/splits --source data/waymo/training_and_validation_and_test --pval 0.10 --ptest 0.03
    
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    parser.add_argument('--pval', required=True,
                        help='proportion of validation data')
    parser.add_argument('--ptest', required=True,
                        help='proportion of test data')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination, args.pval, args.ptest)
    #split('data/waymo/splits', 'data/waymo/training_and_validation_and_test', 0.10, 0.03)

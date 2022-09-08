from typing import Callable, Optional, Tuple, Union
import sys
import numpy as np
import os
from tqdm import tqdm

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

import cv2  # pip install opencv-python # pylint: disable=import-error
import lmdb  # pip install lmdb # pylint: disable=import-error

lmdb_dir = 'ffhq_1024.lmdb'


with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:

    for idx, (_key, value) in tqdm(enumerate(txn.cursor())):

        try:
            try:
                img = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), 1)
                if img is None:
                    raise IOError('cv2.imdecode failed')
                img = img[:, :, ::-1] # BGR => RGB
            except IOError:
                img = np.array(PIL.Image.open(io.BytesIO(value)))

            img_name = _key.decode('utf-8')[-9:-4] + '.png'
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join('lmdb_ffhq',img_name),img)
        except:
            print(sys.exc_info()[1])

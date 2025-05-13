from typing import List

# from cv2.typing import MatLike
from pydantic import BaseModel

# import numpy as np


class Mask(BaseModel):
    matrix: List[List[int]] = []

    # def __init__(self, array: MatLike):
    #     self.matrix = array

    # def tonumpy(self):
    #     return np.array(self.matrix, dtype=np.uint8)

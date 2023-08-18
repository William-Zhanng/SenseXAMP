from typing import List
import numpy as np

def cal_multilabel_alpha(pos_num: List[int], total_samples: int) -> List[float]:
    """
    Calculate alpha to balance positive and negative samples in multilabel classfication
    Args:
        pos_num: positive number of each class.
        total_samples: number of all samples
    Return:
        Alpha number of each class.
    """
    assert total_samples > 0
    pos_num = np.array(pos_num)
    pos_ratio = pos_num / total_samples
    alpha = (1 - pos_ratio)
    return list(alpha)

if __name__ == '__main__':
    # calculate alpha for benchmark2
    pos_num = [15221,5149,1679,6426,2066,2213,2086]
    alpha = cal_multilabel_alpha(pos_num,26878)
    print(alpha)



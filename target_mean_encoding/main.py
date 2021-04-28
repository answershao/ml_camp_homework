# coding = 'utf-8'
import tm
import numpy as np
import pandas as pd
import functools
import time


def count_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        end = time.perf_counter()
        print('{} took {} ms'.format(func.__name__, (end - start) * 1000))
        return res

    return wrapper


@count_time
def cal_target_mean_v1(data):
    return tm.target_mean_v1(data, 'y', 'x')


@count_time
def cal_target_mean_v2(data):
    return tm.target_mean_v2(data, 'y', 'x')


@count_time
def cal_target_mean_v3(data):
    return tm.target_mean_v3(data, 'y', 'x')


@count_time
def cal_target_mean_v4(data):
    return tm.target_mean_v4(data, 'y', 'x')


def main():
    SIZE = 1000
    y = np.random.randint(2, size=(SIZE, 1))
    x = np.random.randint(10, size=(SIZE, 1))
    data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])

    # print(data.head())

    result_1 = cal_target_mean_v1(data)
    result_2 = cal_target_mean_v2(data)
    result_3 = cal_target_mean_v3(data)
    result_4 = cal_target_mean_v4(data)

    diff12 = np.linalg.norm(result_1 - result_2)
    diff13 = np.linalg.norm(result_1 - result_3)
    diff14 = np.linalg.norm(result_1 - result_4)
    print('diff12', diff12)
    print('diff13', diff13)
    print('diff14', diff14)
    diff34 = np.linalg.norm(result_3 - result_4)
    print('diff34', diff34)


if __name__ == "__main__":
    main()

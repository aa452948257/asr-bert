# -*- coding: utf-8 -*-

# @Time    : 2021-12-21 19:31
# @Author  : wangdeyuan

import numpy as np
from math import pi
import matplotlib.pyplot as plt
import matplotlib
import scipy.signal as signal
import math
import json
from tqdm import tqdm

def make_pbsk(input_string:str,snr:int)->str:
    '''
    输入一个汉字和指定的信噪比
    返回这个汉字传输后解码的结果
    信噪比可以为负数，越小错误的概率越高
    '''

    input_num = []
    input_len = [0]
    digit_index = []
    for idx, s in enumerate(input_string):
        if s.isdigit():
            digit_index.append(idx)

    for input_str in input_string:

        b = ord(input_str)

        input_len.append(input_len[-1]+len(bin(b)[2:]))

        for c in bin(b)[2:]:

            input_num.append(c)

    size = len(input_num)

    t = np.arange(0, size)

    for i in range(size):

        if input_num[i] == '1':

            t[i] = 1

        if input_num[i]== '0':

            t[i] = -1

    snr = 10 ** (snr / 10.0)

    npower = 1 / snr

    r = t + np.random.randn(size)*pow(npower, 0.5)

    flag = []

    for i in range(size):

        if r[i]>0:

            flag.append(1)

        else:

            flag.append(0)

    output_str = ''

    for x in range(len(input_len)-1):
        if x not in digit_index:
            output_string = '0b'+''.join([str(i) for i in flag[input_len[x]:input_len[x+1]]])
            output_str = output_str + chr(int(output_string,2))
        else:
            output_str += input_string[x]

    return output_str


def make_sentence_pbsk(input_string, snr):
    output = ''
    for _str in input_string:
        _str = make_pbsk(_str, snr)
        output += _str

    return output


if __name__ == '__main__':
    # begin = '[cls] '
    # sentences = ['查询所有终端的IP', 'Terminal3正常', 'Terminal4关机']
    # end = [' <==> Request-Device_ID-所有终端;Request-Info-IP',
    #        ' <==> ACK-Device_ID-Terminal3;ACK-State-正常',
    #        ' <==> Shutdown-Device_ID-Terminal4']
    #
    # for s in [1, 2]:
    #     sen = sentences[s]
    #     for snr in tqdm(range(11)):
    #         file_name = 's_' + str(s) + '_snr_' + str(snr)
    #         # write_lst = []
    #         with open(file_name, 'w', encoding='utf-8') as f:
    #             for i in tqdm(range(1000)):
    #                 snr_sen = make_sentence_pbsk(sen, snr=snr)
    #                 # write_lst.append(begin + snr_sen + end[s])
    #                 f.write(begin + snr_sen + end[s] + '\n')
    #         f.close()

    string_1 = '2号终端如果SNR小于3，请调整位置到X。'
    # for i in range(1000):
    #     # for j in range(5, 15):
    #     print(make_pbsk(string, -5), 5)
    string_2 = '请全体先切换频段为6，然后上报SNR。'
    for i in range(10):
        print(make_pbsk(string_1, 5))
    for i in range(10):
        print(make_pbsk(string_2, 5))

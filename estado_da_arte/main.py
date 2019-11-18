import sys
import os
import numpy as np
import pandas as pd
import argparse
import includes.tools as tls
from matplotlib import pyplot as plt


def learning(args):
    print('learning!')
    list_models = []
    list_histograms = []
    if os.path.isdir(args.path):
        if args.amount:
            pass
        elif args.models:
            pass
        else:
            list_files = sorted(os.listdir(args.path))
            list_models = [tls.read_obj('{}/{}'.format(args.path, file_name)) for file_name in list_files]
            list_histograms = [tls.Histogram(model) for model in list_models]
            # list_histograms = [tls.Histogram(list_models[0])]
            [tls.write_histogram(hist) for hist in list_histograms]
        pass


def test(args):
    print('test!')
    if not os.path.isdir('hist'):
        print('dir hist not exist. Please try --learning')
        exit(0)
    list_files = os.listdir('hist')
    TOP0 = 0
    TOP1 = 0
    TOP10 = 0
    vet = []
    for file_test_name in list_files:
        compare_list = []
        for file_name in list_files:
            compare_list.append([file_name, tls.hist_compare(tls.read_histogram(file_test_name), tls.read_histogram(file_name))])
        score = pd.DataFrame(compare_list).sort_values(by=1).values[:, 0]
        name = file_test_name.split('_')[1]
        for i in range(1, 11):
            if f'{i}_{name}' == score[0]:
                TOP0 += 1
            if f'{i}_{name}' in score[:2]:
                TOP1 += 1
            if f'{i}_{name}' in score[:10]:
                TOP10 += 1
        cont = 0
        for s in score[:10]:
            if name == s.split('_')[1]:
                cont += 1
        vet.append((cont-1.0) * 100.0/9.0)
    print(f'top0 = {TOP0/40.0*100.0}, top1 = {TOP1/40.0*100.0/2.0}, top10 = {TOP10/40.0*100.0/10.0}, porcentagem media = {sum(vet)/float(len(vet))}')

    # for file_test_name in ['2_Tetraedroobj.dat', '10_Toruloobj.dat']:
    #     compare_list = []
    #     for file_name in list_files:
    #         compare_list.append([file_name, tls.hist_compare(tls.read_histogram(file_test_name), tls.read_histogram(file_name))])
    #     score = pd.DataFrame(compare_list).sort_values(by=1).values[:, 0]
    #     result = score[:10]
    #     print(f'{file_test_name} -> {result}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="models path", type=str)
    parser.add_argument("--amount", help="number of models", type=int)
    parser.add_argument("--models", help="list of model names in quotes",
                        type=str)
    parser.add_argument("--learning", help="if ativate learning",
                        action="store_true")
    parser.add_argument("--test", help="if ativate test", action="store_true")
    args = parser.parse_args()
    if args.learning:
        learning(args)
    if args.test:
        test(args)
    if not (args.learning or args.test):
        print('ops... some args is missing! => try -h for more information XD')

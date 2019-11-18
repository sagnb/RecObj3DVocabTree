import sys
import os
import numpy as np
import pandas as pd
import argparse
from matplotlib import pyplot as plt


sys.path.append('./includes/')


import m3d
import vt
from vv import InvertedFiles


def learning(args):
    print('learning!')
    list_models = []
    list_features = []
    if os.path.isdir(args.path):
        if args.amount:
            pass
        elif args.models:
            pass
        else:
            list_files = sorted(os.listdir(args.path))
            list_models = []
            for file_name in list_files:
                if file_name.split('.')[1] == 'obj':
                    list_models.append(m3d.read_obj('{}/{}'.format(args.path, file_name)))
                else:
                    list_models.append(m3d.read_off('{}/{}'.format(args.path, file_name)))
            list_features = [m3d.write_features(m3d.Features(model)) for model in list_models]


def test(args):
    print('test!')
    inverted_files = InvertedFiles()
    if not os.path.isdir('features'):
        print('dir hist not exist. Please try --learning')
        exit(0)
    list_files = os.listdir('features')
    list_features = []
    for file_name in list_files:
        list_aux , inverted_files = vt.read_features(file_name, path='features/', inverted_files=inverted_files)
        list_features.extend(list_aux)
    tree = vt.Tree()
    # tree.fit(list_features, inverted_files, K=3, L=4, n_docs=len(list_files)) #top0 = 100.0, top1 = 97.22222222222221, top10 = 76.88888888888889, porcentagem media = 74.32098765432096
    tree.fit(list_features, inverted_files, K=3, L=4, n_docs=len(list_files)) #top0 = 100.0, top1 = 98.88888888888889, top10 = 38.88888888888889, porcentagem media = 79.25925925925921
    # tree.fit(list_features, inverted_files, K=2, L=5, n_docs=len(list_files))
    docs_hists = {}
    for file_name in list_files:
        docs_hists[file_name] = tree.get_histogram(file_name)
    TOP0 = 0
    TOP1 = 0
    TOP10 = 0
    vet = []
    for file_test_name in list_files:
        if os.path.isfile('features/{}'.format(file_test_name)):
            compare_list = []
            for file_name in list_files:
                compare_list.append([file_name, vt.hist_compare(docs_hists[file_test_name], docs_hists[file_name])])
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
        else:
            break
    print(f'top0 = {TOP0/40.0*100.0}, top1 = {TOP1/40.0*100.0/2.0}, top10 = {TOP10/40.0*100.0/10.0}, porcentagem media = {sum(vet)/float(len(vet))}')

    # for file_test_name in ['2_Tetraedroobj.dat', '10_Toruloobj.dat']:
    #     if os.path.isfile('features/{}'.format(file_test_name)):
    #         compare_list = []
    #         for file_name in list_files:
    #             compare_list.append([file_name, vt.hist_compare(docs_hists[file_test_name], docs_hists[file_name])])
    #         score = pd.DataFrame(compare_list).sort_values(by=1).values[:, 0]
    #         result = score[:10]
    #         print(f'{file_test_name} -> {result}')

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

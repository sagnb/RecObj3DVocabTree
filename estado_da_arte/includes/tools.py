import numpy as np
import pandas as pd
import os
import math



def hist_compare(hist1, hist2):
    vet_result = []
    for i in range(len(hist1)):
        vet_result.append(hist1[i]/float(np.linalg.norm(hist1)) - hist2[i]/float(np.linalg.norm(hist2)))
    return np.linalg.norm(vet_result)


def kullbackLeiblerDivergenceCriteria(h1, list_histograms):
    print("\nReconhecimento - Kullback Leibler Divergence Criteria")
    print("Reconhecer: {}".format(h1[0]))
    dissi = {}
    for h2 in list_histograms:
        x = 0
        for i in range(1, len(h1)):
            x += (h2[i] - h1[i]) * math.log(h1[i] / h2[i])
        dissi[h2[0]] = - x
    for item in sorted(dissi, key = dissi.get):
        print ("Objeto: " + str(item) + ", Dissimilaridade: " + str(dissi[item]))

def likelihoodCriteria(h1, list_histograms, features):
    print("\nReconhecimento - Likelihood Criteria")
    print("Reconhecer: {}".format(h1[0]))
    dissi = {}
    for Ho in list_histograms:
        x = 0
        for s in range(len(features)):
            a = int((features[s][0]*10)/2.0) % 4 #Alpha
            b = int((features[s][1]*10)/2.0) % 4 #Beta
            c = int((features[s][2]*10)/2.0) % 4 #Gamma
            d = int((features[s][3]*10)/2.0) % 4 #Delta
            ind = (a*(4**3))+(b*(4**2))+(c*(4**1))+(d*(4**0)) #Obtem o indice contiguo
            x += math.log(Ho[ind+1])
        dissi[Ho[0]] = -x
    for item in sorted(dissi, key = dissi.get):
        print ("Objeto: " + str(item) + ", Dissimilaridade: " + str(dissi[item]))


def retrieval(h1, list_histograms, features):
    kullbackLeiblerDivergenceCriteria(h1, list_histograms)
    likelihoodCriteria(h1, list_histograms, features)


def write_histogram(histogram):
    if not os.path.isdir('hist'):
        os.mkdir('hist')
    file = open('hist/{}.dat'.format(histogram.name), 'w')
    file.write('{}\n'.format(histogram.name))
    for i in histogram.histogram:
        for j in i:
            for k in j:
                for l in k:
                    file.write('{}\n'.format(l))
    file.close()


def read_histogram(file_name):
    file = open('hist/{}'.format(file_name))
    hist = []
    # hist.append(file.readline().rstrip('\n'))
    file.readline().rstrip('\n')
    for line in file:
        hist.append(np.float32(line))
    file.close()
    return hist

def read_obj(file_name):
    model = None
    if file_name.split('.')[1] == 'obj':
        file = open(file_name)
        v = []
        f = []

        for line in file:
            if line[0] == 'v':
                v.append(np.float32(line.replace('v', '').split()))
            if line[0] == 'f':
                f.append(np.float32(line.replace('f', '4').split())-1)
        v = pd.DataFrame(v, columns=['x', 'y', 'z'])
        f = pd.DataFrame(f, columns=['numV', 'v1', 'v2', 'v3'])
        model = Model3D(v, f, file_name.split('/')[-1].replace('.', ''))
        file.close()
    return model

def read_off(file_name):
    model = None
    file = open(file_name)
    if file.readline() == 'OFF\n':
        v = []
        f = []
        header = file.readline().split()
        for i in range(int(header[0])):
            v.append(np.float32(file.readline().split()))
        for j in range(int(header[1])):
            f.append(np.int32(file.readline().split()))
        v = pd.DataFrame(v, columns=['x', 'y', 'z'])
        f = pd.DataFrame(f, columns=['numV', 'v1', 'v2', 'v3'])
        model = Model3D(v, f, file_name.split('/')[-1].replace('.', ''))
    file.close()
    return model


def prod_vet(vetA, vetB):
    i = vetA[1] * vetB[2] - vetA[2] * vetB[1]
    j = - vetA[0] * vetB[2] + vetA[2] * vetB[0]
    k = vetA[0] * vetB[1] - vetA[1] * vetB[0]
    return np.float32([i, j, k])


def norma(vet):
    return math.sqrt(math.pow(vet[0],2) + math.pow(vet[1],2) + math.pow(vet[2],2))


def norma_lin(c, features):  # TODO
    minimo = 0
    maximo = 0
    for i in range(len(features)):
        if i == 0:
            minimo = features[0][c]
            maximo = features[0][c]
        else:
            if(features[i][c] < minimo):
                minimo = features[i][c]
            if(features[i][c] > maximo):
                maximo = features[i][c]
    for i in range(len(features)):
        x = features[i][c]
        if (maximo-minimo) == 0.0:
            features[i][c] = maximo/maximo
        else:
            features[i][c] = (x - minimo)/(maximo - minimo)


def get_v(u, p1, p2):
    d = p2 - p1
    nom = prod_vet(d, u)
    den = norma(nom)
    return nom / den if den != 0 else nom


def get_w(u, v):
    return prod_vet(u, v)


def get_alpha(w, u, n):
    x = np.dot(w, n)
    y = np.dot(u, n)
    return math.atan2(y,x)


def get_beta(v, n):
    return np.dot(v, n)


def get_gamma(u, p1, p2):
    d = p2 - p1
    nom = d
    den = norma(nom)
    return np.dot(u, nom / den)


def get_delta(p1, p2):
    d = p2 - p1
    return norma(d)


def get_features(model):
    print(model.name)
    features=[]
    for i, faceI in enumerate(model.faces.values):
        pointsI = [model.vertices.loc[v, :] for v in faceI[1:]]
        compare_xI = pointsI[0].x != pointsI[1].x or pointsI[1].x != pointsI[2].x
        compare_yI = pointsI[0].y != pointsI[1].y or pointsI[1].y != pointsI[2].y
        compare_zI = pointsI[0].z != pointsI[1].z or pointsI[1].z != pointsI[2].z
        if (compare_xI and compare_yI) or (compare_xI and compare_zI) or (compare_yI and compare_xI) or (compare_yI and compare_zI) or (compare_zI and compare_xI) or (compare_zI and compare_yI):
            vetIA = np.float32([pointsI[1].x - pointsI[0].x, pointsI[1].y - pointsI[0].y, pointsI[1].z - pointsI[0].z])
            vetIB = np.float32([pointsI[2].x - pointsI[0].x, pointsI[2].y - pointsI[0].y, pointsI[2].z - pointsI[0].z])
            vet_nI = prod_vet(vetIA, vetIB)
            for j, faceJ in enumerate(model.faces.values):
                # print('{} {}'.format(i, j))
                if i != j and i < j:
                    pointsJ = [model.vertices.loc[v, :] for v in faceJ[1:]]
                    compare_xJ = pointsJ[0].x != pointsJ[1].x or pointsJ[1].x != pointsJ[2].x
                    compare_yJ = pointsJ[0].y != pointsJ[1].y or pointsJ[1].y != pointsJ[2].y
                    compare_zJ = pointsJ[0].z != pointsJ[1].z or pointsJ[1].z != pointsJ[2].z
                    if (compare_xJ and compare_yJ) or (compare_xJ and compare_zJ) or (compare_yJ and compare_xJ) or (compare_yJ and compare_zJ) or (compare_zJ and compare_xJ) or (compare_zJ and compare_yJ):
                        vetJA = np.float32([pointsJ[1].x - pointsJ[0].x, pointsJ[1].y - pointsJ[0].y, pointsJ[1].z - pointsJ[0].z])
                        vetJB = np.float32([pointsJ[2].x - pointsJ[0].x, pointsJ[2].y - pointsJ[0].y, pointsJ[2].z - pointsJ[0].z])
                        vet_nJ = prod_vet(vetJA, vetJB)
                        pointI = None
                        pointJ = None
                        for p1 in pointsI:
                            for p2 in pointsJ:
                                compare_x = p1.x != p2.x
                                compare_y = p1.y != p2.y
                                compare_z = p1.z != p2.z
                                if compare_x or compare_y or compare_z:
                                    pointI = p1
                                    pointJ = p2
                        if type(p1) == type(None):
                            continue
                        d = pointI - pointJ
                        if abs(np.dot(vet_nI, d)) <= abs(np.dot(vet_nJ, d)):
                            u = vet_nI
                            v = get_v(u, pointI, pointJ)
                            w = get_w(u, v)
                            alpha = get_alpha(w, u, vet_nJ)
                            beta = get_beta(v, vet_nJ)
                            gamma = get_gamma(u, pointI, pointJ)
                            delta = get_delta(pointI, pointJ)
                            aux = [alpha, beta, gamma, delta]
                            features.append(aux)
                        else:
                            u = vet_nJ
                            v = get_v(u, pointJ, pointI)
                            w = get_w(u, v)
                            alpha = get_alpha(w, u, vet_nI)
                            beta = get_beta(v, vet_nI)
                            gamma = get_gamma(u, pointJ, pointI)
                            delta = get_delta(pointJ, pointI)
                            aux = [alpha, beta, gamma, delta]
                            features.append(aux)
    # print(features)
    return features


class Model3D(object):
    def __init__(self, vertices, faces, name):
        self.vertices = vertices
        self.faces = faces
        self.name = name

    def __del__(self):
        del self.vertices
        del self.faces


class Histogram(object):
    def __init__(self, model):
        self.name = model.name
        self.histogram = [[[[0 for i in range(4)]for i in range(4)]for i in range(4)]for i in range(4)]
        self.features = None
        if type(model) == Model3D:
            self.features = get_features(model)
            del model
            self.features = np.array(self.features)
            norma_lin(0, self.features)
            norma_lin(1, self.features)
            norma_lin(2, self.features)
            norma_lin(3, self.features)
            for i in range(len(self.features)):
                a = int((self.features[i][0]*10)/2.0) % 4
                b = int((self.features[i][1]*10)/2.0) % 4
                c = int((self.features[i][2]*10)/2.0) % 4
                d = int((self.features[i][3]*10)/2.0) % 4
                self.histogram[a][b][c][d] += 1
            
            acum = 0.0
            for i in range(len(self.histogram)):
                for j in range(len(self.histogram[i])):
                    for k in range(len(self.histogram[i][j])):
                        for l in range(len(self.histogram[i][j][k])):
                            if float(len(self.features)) != 0.0:
                                self.histogram[i][j][k][l] = (self.histogram[i][j][k][l]/float(len(self.features)))
                            acum += self.histogram[i][j][k][l]
            print("A soma do histogram eh: ", acum)
            menorBin = 0.0
            flag = 0
            for i in range(len(self.histogram)):
                for j in range(len(self.histogram[i])):
                    for k in range(len(self.histogram[i][j])):
                        for l in range(len(self.histogram[i][j][k])):
                            if flag == 0:
                                if self.histogram[i][j][k][l] != 0.0:
                                    menorBin = self.histogram[i][j][k][l]
                                    flag = 1
                            else:
                                if (self.histogram[i][j][k][l] < menorBin) and (self.histogram[i][j][k][l] != 0.0):
                                    menorBin = self.histogram[i][j][k][l]
            for i in range(len(self.histogram)):
                for j in range(len(self.histogram[i])):
                    for k in range(len(self.histogram[i][j])):
                        for l in range(len(self.histogram[i][j][k])):
                            if self.histogram[i][j][k][l] == 0.0:
                                self.histogram[i][j][k][l] = menorBin/2


if __name__ == '__main__':
    pass

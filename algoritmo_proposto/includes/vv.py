'''
Visual Vocabulary
'''


# class BOW(dict):

#     # __slots__


# class StopList(list):

#     # __slots__


# def TF_IDF(word, bow):
#     # TODO
#     TFp = None
#     D = len(bow.keys())
#     Dp = float(bow[word])
#     IDFp = np.log(D/Dp)


class InvertedFiles(object):

    __slots__ = 'INDEX'

    def __init__(self):
        self.INDEX = {}

    def __del__(self):
        del self.INDEX

    def word_file(self, word, file):
        if not word in self.INDEX:
            self.INDEX[word] = []
        if not file in self.INDEX[word]:
            self.INDEX[word].append(file)


if __name__ == "__main__":
    pass

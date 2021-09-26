from random import Random
import numpy as np
import torch
import pickle
import re, inspect, json
from enum import Enum

class Universal(object):
    r = Random()

    @classmethod
    def getrand(cls, seed):
        cls.r.seed(seed)
        return cls.r.random()

    @classmethod
    def sample(cls, data):
        r = cls.r.random()
        ll = len(data)
        s = 0
        d_tensor = torch.tensor(data, dtype=torch.float32)
        d_soft = torch.softmax(d_tensor, dim=0)
        for index, v in enumerate(d_soft.numpy()):
            s += v
            if r < s:
                return index

    @staticmethod
    def saveClass(obj:object, path, filt=[]):
        return Universal.saveClassWithJSON(obj, path, filt=filt)

    @staticmethod
    def loadClass(obj, path):
        return Universal.loadClassWithJSON(obj, path)

    @staticmethod
    def saveClassWithJSON(obj:object, path, filt =[]):
        pattern = re.compile('_+.*')
        property_list = [i for i in dir(obj) if not (re.match(pattern, i)
                                                     or inspect.ismethod(getattr(obj, i))
                                                     or inspect.isfunction(getattr(obj,i))
                                                     or type(getattr(obj, i)) is type(Enum('A',('a', )))
                                                     or i in filt)]
        result = {}
        for pn in property_list:
            try:
                result[pn] = json.dumps(getattr(obj, pn), ensure_ascii=False)
            except TypeError as e:
                print(e)
        result_json = json.dumps(result, ensure_ascii=False)
        filename = obj.__name__
        filepath = '{}/{}.json'.format(path, filename)
        fq = open(filepath, 'w', encoding='utf-8')
        fq.write(result_json)
        fq.close()
        pass

    @staticmethod
    def loadClassWithJSON(obj:object, path):
        fq = open(path, 'r', encoding="utf-8")
        result_json = fq.read()
        fq.close()
        result = json.loads(result_json)
        for k,v in result.items():
            v = json.loads(v)
            setattr(obj, k, v)
        return obj

    @staticmethod
    def saveClassWithPickle(obj:object, path):
        filename = obj.__name__
        filepath = '{}/{}.pickle'.format(path, filename)
        fq = open(filepath, 'wb')
        pickle.dump(obj, fq)
        fq.close()
        pass

    @staticmethod
    def loadClassWithPickle(path):
        fq = open(path, 'rb')
        result = pickle.load(fq)
        fq.close()
        return result

# import pandas as pd
# def savefile():
#     data = {
#         'a': list(range(10)),
#         'b': list(range(10)),
#         'c': list(range(10)),
#         'd': list(range(10)),
#         'e': list(range(10)),
#         'a': list(range(10)),
#         'b': list(range(10)),
#     }
#     df = pd.DataFrame(data)
#     df.to_csv(path_or_buf='./2.csv', index=None)
#     pass
#
# if __name__ == '__main__':
#     savefile()
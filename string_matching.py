import numpy as np
import os
import pickle
import re
from tqdm import tqdm
import time
import nmslib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List


class StringMatching:
    def __init__(self, data_or_path, load_from_path=False):
        if load_from_path:
            self._load(data_or_path)
        else:
            if not isinstance(data_or_path, list):
                raise ValueError("Expected list of strings to be passed")
            self.data = data_or_path
            self.vectorizer = TfidfVectorizer(min_df=1, analyzer=self._ngrams)
            self.tf_idf_matrix = self.vectorizer.fit_transform(self.data)
            self._save()
        # implement index saving/loading
        self.index = self._create_index()
    
    def _save(self):
        try:
            os.mkdir("artifacts")
        except FileExistsError:
            pass
        pickle.dump(self.vectorizer, open("artifacts/vectorizer.pkl", "wb"))
        pickle.dump(self.tf_idf_matrix, open("artifacts/tf_idf_matrix.pkl", "wb"))
        pickle.dump(self.data, open("artifacts/data.pkl", "wb"))
        
        
    def _load(self, path): #error handling
        self.vectorizer = pickle.load(open(os.path.join(path, "vectorizer.pkl"),"rb"))
        self.tf_idf_matrix = pickle.load(open(os.path.join(path, "tf_idf_matrix.pkl"),"rb"))
        self.data = pickle.load(open(os.path.join(path, "data.pkl"),"rb"))
        
        
    def _ngrams(self, string, n=3):
        string = str(string)
        string = string.lower() # lower case
        string = string.replace('&', 'and')
        string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single
        string = ' '+ string +' ' # pad names for ngrams...
        ngrams = zip(*[string[i:] for i in range(n)])
        return [''.join(ngram) for ngram in ngrams]
    
    
    def _create_index(self):
        index = nmslib.init(method='simple_invindx', space='negdotprod_sparse_fast', data_type=nmslib.DataType.SPARSE_VECTOR) 
        index.addDataPointBatch(self.tf_idf_matrix)
        start = time.time()
        index.createIndex() 
        end = time.time() 
        print(f"Indexing time = {end-start}")
        return index
    
            
    def query(self, q: List[str], K=1, num_threads=4):
        if isinstance(q, str):
            q = [q]
            num_threads = 1
        query_matrix = self.vectorizer.transform(q)
        query_qty = query_matrix.shape[0]
        start = time.time()
        nbrs = self.index.knnQueryBatch(query_matrix, k = K, num_threads = num_threads)
        end = time.time() 
#         print(f"kNN time total={end-start} (sec), per query={float(end-start)/query_qty} (sec)," \
#         f"per query adjusted for thread number={num_threads*float(end-start)/query_qty} (sec)")
        return self._parse_results(q, nbrs, K)
        
    
    def _parse_results(self, q, res, k):
        out = []
        for i, q_i in enumerate(q):
            out.append({
                "entity": q_i,
                "match": [self.data[res[i][0][j]] if res[i][0].shape[0] > j else '' for j in range(k)],
                "score": [res[i][1][j] if res[i][1].shape[0] > j else 0.0 for j in range(k)]
            })
        return out

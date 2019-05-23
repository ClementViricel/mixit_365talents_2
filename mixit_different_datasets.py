import pandas as pd
from gensim.models import KeyedVectors

# KeyedVectors doc : https://radimrehurek.com/gensim/models/keyedvectors.html


w2v_ngrams_TD = KeyedVectors.load("models/WV_TD.wv")
w2v_ngrams_con = KeyedVectors.load("models/WV_con.wv")
w2v_ngrams_pol = KeyedVectors.load("models/WV_pol.wv")


def compare(tokens, wvs, titles):
   results = {}
   for wv, title in zip(wvs, titles):
       results.update({title: [i[0] for i in wv.most_similar(tokens)]})
   return pd.DataFrame(data=results)


compare("text", [w2v_ngrams_TD, w2v_ngrams_con, w2v_ngrams_pol], ["The_Donald", "conspiracy", "politics"])





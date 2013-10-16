from pyspark import SparkContext

import json
import time
import math
from operator import add

print 'loading'
sc = SparkContext("spark://ec2-107-22-0-110.compute-1.amazonaws.com:7077", "TF-IDF", pyFiles=['term_tools.py'])

from term_tools import get_terms

corpus = sc.textFile('s3n://AKIAJFDTPC4XX2LVETGA:lJPMR8IqPw2rsVKmsSgniUd+cLhpItI42Z6DCFku@6885public/enron/lay-k.json')
#corpus = sc.textFile('s3n://AKIAJFDTPC4XX2LVETGA:lJPMR8IqPw2rsVKmsSgniUd+cLhpItI42Z6DCFku@6885public/fsosa/short.json')

json_corpus = corpus.map(lambda x: json.loads(x)).cache()

# Calculate per-term idf
term_counts = json_corpus.flatMap(lambda x: get_terms(x['text'])).map(lambda y: (y, 1)).reduceByKey(add)
per_term_idf = term_counts.map(lambda x: (x[0], math.log(516893.0 / x[1])))
output = per_term_idf.collect()
for x in output: 
  print x






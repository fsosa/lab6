from pyspark import SparkContext

import json
import time
import math
from operator import add

print 'loading'
sc = SparkContext("spark://ec2-107-22-0-110.compute-1.amazonaws.com:7077", "TF-IDF", pyFiles=['term_tools.py'])

from term_tools import get_terms

def sender_term_pairs(email):
  sender = email['sender']
  return map(lambda x: {'sender': sender, 'term': x}, get_terms(email['text']))


def single_sender_term_freq(sender_list):
  sender_tf = {}
  for pair in sender_list:
    key = pair['sender']
    if (key in sender_tf):
        sender_tf[key] += 1
    else:
        sender_tf[key] = 1

  return sender_tf

#(u'talk', [{'term': u'talk', 'sender': u'rosalee.fleming@enron.com'}, {'term': u'talk', 'sender': u'rosalee.fleming@enron.com'}, {'term': u'talk', 'sender': u'rosalee.fleming@enron.com'}])
  
def sender_tf(grouped_pair):
  # Calculate the term frequency for the term-grouped pair
  tf_dict = single_sender_term_freq(grouped_pair[1])
  term = grouped_pair[0]

  return map(lambda y: (term, y), tf_dict.items())


#corpus = sc.textFile('s3n://AKIAJFDTPC4XX2LVETGA:lJPMR8IqPw2rsVKmsSgniUd+cLhpItI42Z6DCFku@6885public/enron/lay-k.json')
corpus = sc.textFile('s3n://AKIAJFDTPC4XX2LVETGA:lJPMR8IqPw2rsVKmsSgniUd+cLhpItI42Z6DCFku@6885public/fsosa/short.json')

json_corpus = corpus.map(lambda x: json.loads(x)).cache()

# Calculate per-term idf
term_counts = json_corpus.flatMap(lambda x: get_terms(x['text'])).map(lambda y: (y, 1)).reduceByKey(add)
per_term_idf = term_counts.map(lambda x: (x[0], math.log(516893.0 / x[1]))).cache()

# Get sender/term pairs
grouped_sender_term_pairs = json_corpus.flatMap(sender_term_pairs).groupBy(lambda x: x['term'])

# Calculate sender-term frequency
sender_tf = grouped_sender_term_pairs.flatMap(sender_tf)

output = sender_tf.collect()
for x in output: 
   print x

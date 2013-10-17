from pyspark import SparkContext

import json
import time
import math
from operator import add
from difflib import get_close_matches

print 'loading'
sc = SparkContext("spark://ec2-107-22-0-110.compute-1.amazonaws.com:7077", "TF-IDF", pyFiles=['term_tools.py'])

from term_tools import get_terms

# Returns a list of dicts => each dict containing a sender and a term
def sender_term_pairs(email):
  sender = email['sender']
  master_email = sender

  if sender in email_to_master:
    master_email = email_to_master[sender]

  return map(lambda x: {'sender': master_email, 'term': x}, get_terms(email['text']))

# [{'term': u'talk', 'sender': u'rosalee.fleming@enron.com'}, {'term': u'talk', 'sender': u'rosalee.fleming@enron.com'}, {'term': u'talk', 'sender': u'rosalee.fleming@enron.com'}]
#
# Given a list like the one above,  creates a dictionary of the frequency of the term by the sender 
def single_sender_term_freq(sender_list):
  sender_tf = {}
  for pair in sender_list:
    key = pair['sender']
    if (key in sender_tf):
        sender_tf[key] += 1
    else:
        sender_tf[key] = 1

  return sender_tf
 
# Given a tuple of a term and a list of individual occurences by different senders, calculates the term frequency per sender
def sender_tf(grouped_pair):
  tf_dict = single_sender_term_freq(grouped_pair[1])
  term = grouped_pair[0]

  return map(lambda y: (term, y), tf_dict.items())


def consolidate_emails(grouped_pair):
  letter = grouped_pair[0]
  email_pairs = grouped_pair[1]
  all_names = [x[1] for x in email_pairs]
  name_email_dict = {}
  
  for x in email_pairs:
    name_email_dict[x[1]] = x[0]


  consolidated_emails = []
  for pair in email_pairs:
    email = pair[0]
    name = pair[1]
    close_matches = get_close_matches(name, all_names, cutoff=0.8)
    matching_emails = []
    for x in close_matches:
       matching_emails.append(name_email_dict[x])

    consolidated_emails.append((email, set(matching_emails)))
    
  return map(lambda y: y, consolidated_emails)



#---- BEGIN PROCESSING ------#
corpus = sc.textFile('s3n://AKIAJFDTPC4XX2LVETGA:lJPMR8IqPw2rsVKmsSgniUd+cLhpItI42Z6DCFku@6885public/enron/lay-k.json')
#corpus = sc.textFile('s3n://AKIAJFDTPC4XX2LVETGA:lJPMR8IqPw2rsVKmsSgniUd+cLhpItI42Z6DCFku@6885public/fsosa/short.json')

json_corpus = corpus.map(lambda x: json.loads(x)).cache()


#--- Disambiguation ---#
# Approach: 
# - Get a list of distinct emails from the data set
# - Sort them into groups based on the first letter of the name part of the email (i.e. before '@')
# - Create a dictionary of email to master where master is the email that we will consolidate our searches over
#     e.g ken.lay@enron.com and ken.lay@yahoo.com have as their master email ken.lay@enron.com 
# - Broadcast to all nodes so that they know about it
unique_emails = json_corpus.map(lambda x: x['sender']).distinct()
lastnames = unique_emails.map(lambda x: (x,x.split("@")[0])).groupBy(lambda x: x[1][0]) 
consolidated = lastnames.flatMap(consolidate_emails).flatMap(lambda x: map(lambda y: (y, x[0]), x[1])).collectAsMap()
email_to_master = sc.broadcast(consolidated)

#----- Actual TF-IDF Calculation -----#
# Calculate per-term idf
term_counts = json_corpus.flatMap(lambda x: get_terms(x['text'])).map(lambda y: (y, 1)).reduceByKey(add)
per_term_idf = term_counts.map(lambda x: (x[0], math.log(516893.0 / x[1]))).cache()

# Get sender/term pairs
grouped_sender_term_pairs = json_corpus.flatMap(sender_term_pairs).groupBy(lambda x: x['term'])

# Calculate sender-term frequency
sender_tf = grouped_sender_term_pairs.flatMap(sender_tf).cache()

#e.g. join: (u'talk', ((u'rosalee.fleming@enron.com', 3), 12.056978880153091))
tfidf = sender_tf.join(per_term_idf).map(lambda x:{'sender': x[1][0][0], 'term':x[0], 'tf-idf':x[1][0][1]*x[1][1]})

output = tfidf.collect()
for x in output:
  print x

import random
import csv
import math
import pandas
import codecs
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
from IPython.display import HTML, display

SMOOTH_CONST = 0.001 # we want this to be smaller than 1/n where n is the size of the largest training category. that way, any word that has appeared exactly once (with category c) in training will still have a larger probability for category c, than any other category c'
TRAIN_SPLIT = 0.8

categories = ['Energy', 'Food', 'Medical', 'None', 'Water']
need_or_resource_labels = ['need', 'resource', 'N/A']


class Tweet(object):
  def __init__(self, tweetSurfaceForm, category, need_or_resource):
    if isinstance(tweetSurfaceForm, unicode):
      self.tweetTokens = word_tokenize(tweetSurfaceForm)
    else:
      self.tweetTokens = word_tokenize(tweetSurfaceForm.decode('utf-8','ignore'))
    self.category = category
    self.need_or_resource = need_or_resource

  def __getitem__(self,index):
    return self.tweetTokens[index]

  def idx(self, token):
    return self.tweetTokens.index(token)

  def __unicode__(self):
    return " ".join(self.tweetTokens)

  def __str__(self):
    return unicode(self).encode('utf-8')

  def __repr__(self):
      return self.__str__()


def read_data(path = 'data/labeled-data-singlelabels.csv'):
  """Returns two lists of tweets: the train set and the test set"""
  data = {}
  f = codecs.open(path, encoding='utf-8')
  with open(path) as f:
    reader = csv.reader(f)
    for row in reader:
      (tweetId, tweetText, category, need_or_resource) = row
      assert category in categories
      assert need_or_resource in need_or_resource_labels
      if need_or_resource == "N/A":
        assert category == "None"
      assert tweetId not in data.keys()
      data[tweetId] = Tweet(tweetText, category, need_or_resource)

  data = data.values() # list of Tweets

  # for c in categories:
  #   print "%i tweets with category %s" % (len([d for d in data if d.category==c]), c)
  # for n in needs:
  #   print "%i tweets with need/resource %s" % (len([d for d in data if d.need_or_resource==n]), n)

  train_size = int(len(data) * TRAIN_SPLIT)
  random.seed(7)
  random.shuffle(data)
  train_tweets = data[:train_size]
  test_tweets = data[train_size:]
  for c in categories:
    assert len([t for t in test_tweets if t.category==c])>=10
    assert len([t for t in train_tweets if t.category==c])>=10

  # while True:
  #   random.shuffle(data)
  #   train_tweets = data[:train_size]
  #   test_tweets = data[train_size:]
  #   try:
  #     for c in categories:
  #       assert len([t for t in test_tweets if t.category==c])>=10
  #       assert len([t for t in train_tweets if t.category==c])>=10
  #   except AssertionError:
  #     continue # shuffle and try again
  #   break # break out of while True loop and return data

  # print "Split into %i training and %i test tweets\n" % (len(train_tweets), len(test_tweets))
  return train_tweets, test_tweets


def featurize(tweet):
  return set([t.lower() for t in tweet.tweetTokens])


def show_confusion_matrix(predictions):
  """Displays a confusion matrix as a HTML table.
  Rows are true label, columns are predicted label.
  predictions is a list of (tweet, predicted_category) pairs"""
  num_categories = len(categories)
  conf_mat = np.zeros((num_categories, num_categories), dtype=np.int32)
  for (tweet,predicted_category) in predictions:
    gold_idx = categories.index(tweet.category)
    predicted_idx = categories.index(predicted_category)
    conf_mat[gold_idx, predicted_idx] += 1
  df = pandas.DataFrame(data=conf_mat, columns=categories, index=categories)
  display(HTML(df.to_html()))


def class2color_style(s):
  class2color = {
    'Energy' : 'red',
    'Food': 'orange',
    'Medical': 'green',
    'None': 'gray',
    'Water': 'blue',
    'resource': 'purple',
    'need': 'pink',
    'N/A': 'gray',
  }
  try:
    return "color: %s" % class2color[s]
  except KeyError:
    return "color: black"


def show_tweets(tweets, search_term=None):
  """Displays a HTML table of tweets alongside labels"""
  if search_term is not None:
    tweets = [t for t in tweets if search_term in str(t).lower()]
  columns = ['Text', 'Category', 'Need or resource']
  data = [[unicode(t), t.category, t.need_or_resource] for t in tweets]
  pandas.set_option('display.max_colwidth', -1)
  df = pandas.DataFrame(data, columns=columns)
  s = df.style.applymap(class2color_style)\
              .set_properties(**{'text-align': 'left'})
  display(HTML(s.render()))


def show_predictions(predictions, show_mistakes_only=False):
  """Displays a HTML table comparing true categories to predicted categories.
  predictions is a list of (tweet, predicted_category) pairs"""
  if show_mistakes_only:
    predictions = [(t,p) for (t,p) in predictions if t.category!=p]
  columns = ['Text', 'True category', 'Predicted category']
  data = [[unicode(t), t.category, predicted_category] for (t,predicted_category) in predictions]
  pandas.set_option('display.max_colwidth', -1)
  df = pandas.DataFrame(data, columns=columns)
  s = df.style.applymap(class2color_style)\
              .set_properties(**{'text-align': 'left'})
  display(HTML(s.render()))


def get_category_prob(tweet, prob_c, feature_probs_c):
    """Calculate P(c|tweet) for a category c"""
    ans = prob_c
    for x in featurize(tweet):
      if x in feature_probs_c.keys():
        ans *= feature_probs_c[x]
      else:
        ans *= SMOOTH_CONST
    return ans


def most_discriminative(tweets, feature_probs, category_probs):
  """Prints, for each category, which features are most discriminative i.e. maximize P(category|feature), including normalization by P(feature)"""
  all_features = set([feature for tweet in tweets for feature in featurize(tweet)])

  feat2dist = {} # maps feature f to a probability distribution over categories, for a tweet containing just this feature

  for f in all_features:
    single_feature_tweet = Tweet(f, "", "")
    dist = {c: get_category_prob(single_feature_tweet, category_probs[c], feature_probs[c]) for c in categories}
    s = sum([dist[c] for c in categories])
    dist = {c: dist[c]/s for c in categories}
    feat2dist[f] = dist

  # for each category print the features that maximize P(C|f) (normalized by P(f))
  print "MOST DISCRIMINATIVE FEATURES: \n"
  for c in categories:
    probs = [(f,dist[c]) for f,dist in feat2dist.iteritems()]
    probs = sorted(probs, key=lambda x: x[1], reverse=True)
    print "{0:20} {1:10}".format("Feature", "P(%s|feature)"%c)
    for (f,p) in probs[:10]:
        print "{0:20} {1:.4f}".format(f,p)
    print ""

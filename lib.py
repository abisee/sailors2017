import random
import csv
import numpy as np
from collections import Counter
import pandas
import codecs
from nltk.tokenize import word_tokenize
import math
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

  def __unicode__(self):
    return " ".join(self.tweetTokens)

  def __str__(self):
    return unicode(self).encode('utf-8')

  def __repr__(self):
      return self.__str__()


def get_entropy(feature, class_probs, feature_probs):
  p_class_given_feature = [prob_class_given_features(c, [feature], class_probs, feature_probs) for c in sorted(class_probs.keys())] # list of p(class | feature) for each class
  base_dist = [class_probs[c] for c in sorted(class_probs.keys())] # p(class) for each class
  assert sum(base_dist)==1.0, "%.10f" % (sum(base_dist)-1)
  p_class_given_feature = [p/sum(p_class_given_feature) for p in p_class_given_feature]
  cross_entropy = -sum([p1*math.log(p2) for (p1,p2) in zip(p_class_given_feature, base_dist)])
  return cross_entropy, p_class_given_feature


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


def token_seen(token, feature_probs):
  return any([token in feat2count.keys() for feat2count in feature_probs.values()])


def prob_class_given_features(c, features, class_probs, feature_probs):
  """Get P(class|features)"""
  prob = class_probs[c]
  for f in features:
    try:
      prob *= feature_probs[c][f]
    except KeyError: # haven't seen feature f in training data
      prob *= SMOOTH_CONST
  return prob


def classify_nb(tweet, class_probs, feature_probs):
  # print "classifying: ", tweet
  features = featurize(tweet)
  if all([not token_seen(f,feature_probs) for f in features]):
    print "WARNING: all unseen tokens: ", tweet

  probs = {c: prob_class_given_features(c, features, class_probs, feature_probs) for c in class_probs.keys()}
  predicted_class = max(probs.iterkeys(), key=(lambda l: probs[l]))
  return predicted_class


# def classify_rb(tweet):
#   tweet_text = str(tweet).lower()
#   if "food" in tweet_text or "hungry" in tweet_text or "meal" in tweet_text or "snacks" in tweet_text or "eat" in tweet_text:
#     return 'Food'
#   elif "medical" in tweet_text or "medicine" in tweet_text or "injury" in tweet_text or "hospital" in tweet_text:
#     return 'Medical'
#   elif "energy" in tweet_text or "power" in tweet_text or "electricity" in tweet_text:
#     return 'Energy'
#   elif "water" in tweet_text or "thirsty" in tweet_text or "drink" in tweet_text:
#     return 'Water'
#   else:
#     return 'None'



def learn_nb(tweets):
  """Returns class_probs and feature_probs.
  class_probs maps class name to probability.
  feature_probs maps class name to a dictionary mapping feature names to p(feature|class).
  """
  num_tweets = len(tweets)

  class_counts = Counter()
  for tweet in tweets:
    class_counts[tweet.category] += 1
  print "training class counts: ", class_counts

  categories = sorted(class_counts.keys())

  class_probs = {label: float(count)/float(num_tweets) for label,count in class_counts.iteritems()}
  print "\nclass probs: "
  for l,p in class_probs.iteritems():
    print "%s: %.2f" % (l,p)

  all_features = []
  feature_only_counts = Counter()

  feature_counts = {label: Counter() for label in class_counts.keys()} # maps label to a Counter that maps feature -> count
  for tweet in tweets:
    features = featurize(tweet)
    label = tweet.category
    for f in features:
      feature_counts[label][f] += 1
      feature_only_counts[f] += 1
      if f not in all_features: all_features.append(f)

  feature_only_probs = {f : float(c)/float(num_tweets) for f,c in feature_only_counts.iteritems()} # gives p(feature) for each feature

  print "\nmost common features: "
  for f,p in feature_only_counts.most_common(10):
    print f,p
  print ""

  feature_probs = {
    label: {
      feature: float(count)/float(class_counts[label]) for feature,count in feat2count.iteritems()
    } for label, feat2count in feature_counts.iteritems()
  }

  # for c,feat2count in feature_counts.iteritems():
  #   print ""
  #   print "top features for class %s:" % c
  #   for (feat, count) in feat2count.most_common(10):
  #     print feat, count
  #   print ""

  # for c in categories:
  #   diffs = []
  #   for f in all_features:
  #     try:
  #       c_prob = feature_probs[c][f]
  #     except KeyError:
  #       c_prob = SMOOTH_CONST
  #     prior = feature_only_probs[f]
  #     diffs.append((f,c_prob-prior,c_prob,prior)) # difference between p(feature|class) and p(feature)
  #   diffs = sorted(diffs, key=lambda x: x[1], reverse=True)
  #   print "top features for class %s" % c
  #   for (feat, diff, c_prob, prior) in diffs[:10]:
  #     print "%s, %.2f (%.2f -> %.2f)" % (feat, diff, prior, c_prob)
  #   print ""

  feat2entropy = [(feature, get_entropy(feature, class_probs, feature_probs)) for feature in all_features]
  feat2entropy = sorted(feat2entropy, key=lambda x: x[1][0], reverse=True) # sort by entropy, descending
  print "Words with highest cross entropy:"
  for (f,(e,prob_categories_given_feature)) in feat2entropy[:10]:
    print "%s: entropy %.2f, " % (f,e),
    for c_idx,c in enumerate(sorted(class_counts.keys())):
      print "%s: %.4f, " % (c, prob_categories_given_feature[c_idx]),
    print ""
  print "\nWords with lowest cross entropy:"
  for (f,(e,prob_categories_given_feature)) in feat2entropy[-10:]:
    print "%s: entropy %.2f, " % (f,e),
    for c_idx,c in enumerate(sorted(class_counts.keys())):
      print "%s: %.4f, " % (c, prob_categories_given_feature[c_idx]),
    print ""

  feat2p_c_given_f = {k:v[1] for (k,v) in feat2entropy}
  for c_idx,c in enumerate(categories):
    c_prob = class_probs[c]
    feat2p_c_given_f_lst = [(f, feat2p_c_given_f[f][c_idx]) for f in all_features]
    feat2p_c_given_f_lst = sorted(feat2p_c_given_f_lst, key=lambda x: x[1], reverse=True)
    print "\nTop features for %s which has base prob %.4f" % (c, c_prob)
    for (f, p_c_given_f) in feat2p_c_given_f_lst[:10]:
      print "%s: %.4f (%i/%i occurrences)" % (f, p_c_given_f, feature_counts[c][f], feature_only_counts[f]) # features that maximize p(C|f)
    print ""


  return class_probs, feature_probs

def evaluate(tweet2prediction):
  # get unbalanced accuracy
  # num_correct = len([1 for tweet,predicted_label in tweet2prediction.iteritems() if tweet.category==predicted_label])
  # total = len(tweet2prediction.keys())
  # acc = float(num_correct)*100/float(total)
  # print "%i correct of %i. Unbalanced accuracy = %.2f percent\n" % (num_correct, total, acc)

  # get recall / precision / F1 per class
  class2f1 = {}
  for c in categories:
    num_correct = len([1 for tweet,predicted_label in tweet2prediction.iteritems() if tweet.category==c and predicted_label==c])

    num_fp = len([1 for tweet,predicted_label in tweet2prediction.iteritems() if tweet.category!=c and predicted_label==c])

    num_fn = len([1 for tweet,predicted_label in tweet2prediction.iteritems() if tweet.category==c and predicted_label!=c])

    if num_correct==0:
      precision = 0.0
      recall = 0.0
      f1 = 0.0
    else:
      precision = float(num_correct)*100 / float(num_correct + num_fp)
      recall = float(num_correct)*100 / float(num_correct + num_fn)
      f1 = 2*precision*recall / (precision+recall)

    print "Class %s: precision %.2f, recall %.2f, F1 %.2f" % (c, precision, recall, f1)
    class2f1[c]=f1
  balanced_f1 = sum(class2f1.values())/len(class2f1)
  print "Balanced F1: %.2f percent\n" % (balanced_f1)


def show_confusion_matrix(predictions):
  """Prints confusion matrix. rows are gold label, columns are predicted label."""
  num_categories = len(categories)
  conf_mat = np.zeros((num_categories, num_categories), dtype=np.int32)
  for (tweet,predicted_label) in predictions:
    gold_idx = categories.index(tweet.category)
    predicted_idx = categories.index(predicted_label)
    conf_mat[gold_idx, predicted_idx] += 1

  df = pandas.DataFrame(data=conf_mat, columns=categories, index=categories)
  display(HTML(df.to_html()))


def classify_unseen_nb(test_tweets, class_probs, feature_probs):
  tweet2prediction = {tweet: classify_nb(tweet, class_probs, feature_probs) for tweet in test_tweets} # maps each unseen tweet to its predicted label
  print "\nEvaluating NB system on test set..."
  evaluate(tweet2prediction)
  show_confusion_matrix(tweet2prediction)


def classify_unseen_rb(test_tweets):
  tweet2prediction = {tweet: classify_rb(tweet) for tweet in test_tweets} # maps each unseen tweet to its predicted label
  print "\nEvaluating rule-based system on test set..."
  evaluate(tweet2prediction)
  show_confusion_matrix(tweet2prediction)


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
  if search_term is not None:
    tweets = [t for t in tweets if search_term in str(t).lower()]
  columns = ['Text', 'Category', 'Need or resource']
  data = [[unicode(t), t.category, t.need_or_resource] for t in tweets]
  pandas.set_option('display.max_colwidth', -1)
  df = pandas.DataFrame(data, columns=columns)
  s = df.style.applymap(class2color_style)
  display(HTML(s.render()))


def show_predictions(predictions, show_mistakes_only=False):
  """predictions is a list of (tweet, predicted_class) pairs"""
  if show_mistakes_only:
    predictions = [(t,p) for (t,p) in predictions if t.category!=p]
  columns = ['Text', 'True category', 'Predicted category']
  data = [[unicode(t), t.category, predicted_class] for (t,predicted_class) in predictions]
  pandas.set_option('display.max_colwidth', -1)
  df = pandas.DataFrame(data, columns=columns)
  s = df.style.applymap(class2color_style)
  display(HTML(s.render()))


# if __name__ == "__main__":
#   train_tweets, test_tweets, categories, needs = read_data()
#
#   # for t in train_tweets:
#   #   if ".." in featurize(t):
#   #     print t.category
#   #     print t
#   #     print ""
#
#   class_probs, feature_probs = learn_nb(train_tweets)
#   classify_unseen_nb(test_tweets, class_probs, feature_probs)
#   classify_unseen_rb(test_tweets)

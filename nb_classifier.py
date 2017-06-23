import random
import csv
import numpy as np
from collections import Counter, defaultdict
import codecs
from nltk.tokenize import word_tokenize
import math
# import scipy

SMOOTH_CONST = 0.001 # we want this to be smaller than 1/n where n is the size of the largest training category. that way, any word that has appeared exactly once (with class c) in training will still have a larger probability for class c, than any other class c'
TRAIN_SPLIT = 0.8


class Tweet:
  def __init__(self, tweetSurfaceForm, category, need_or_resource):
    if isinstance(tweetSurfaceForm, unicode):
      self.tweetTokens = word_tokenize(tweetSurfaceForm)
    else:
      self.tweetTokens = word_tokenize(tweetSurfaceForm.decode('utf-8','ignore'))
    self.category = category
    self.need_or_resource = need_or_resource # this can be "need", "resource" "-" (if category is None) or "" (if talking about a not-None category but not giving a need or resource)

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
  assert sum(base_dist)==1
  p_class_given_feature = [p/sum(p_class_given_feature) for p in p_class_given_feature]
  cross_entropy = -sum([p1*math.log(p2) for (p1,p2) in zip(p_class_given_feature, base_dist)])
  return cross_entropy, p_class_given_feature


def read_data(path = 'data/labeled-data.csv'):
  """Returns two lists of tweets: the train set and the test set"""

  data = defaultdict(list)
  f = codecs.open(path, encoding='utf-8')
  with open(path) as f:
    reader = csv.reader(f)
    for row in reader:
      # print row
      (tweetId, tweetText, category, need_or_resource) = row
      # print tweetId
      data[tweetId].append(Tweet(tweetText, category, need_or_resource))
  print "read %i rows from file" % len(data)

  num_mult = sum([1 if len(v)>1 else 0 for v in data.values()])
  print "%i of the tweets have multiple labels" % num_mult

  data = {k:v for k,v in data.iteritems() if len(v)==1}

  for v in data.values(): assert len(v)==1
  data = [v[0] for v in data.values()]
  print "removed those with multiple labels. now have %i tweets" % len(data)

  categories = sorted(list(set([t.category for t in data])))
  needs = sorted(list(set([t.need_or_resource for t in data])))
  print categories
  print needs

  for c in categories:
    print "%i tweets with category %s" % (len([d for d in data if d.category==c]), c)
  for n in needs:
    print "%i tweets with need/resource %s" % (len([d for d in data if d.need_or_resource==n]), n)

  random.seed(7)
  random.shuffle(data)

  train_size = int(len(data) * TRAIN_SPLIT)
  train_tweets = data[:train_size]
  test_tweets = data[train_size:]
  print "split into %i training and %i test tweets\n" % (len(train_tweets), len(test_tweets))
  return train_tweets, test_tweets, categories, needs


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


def classify_rb(tweet):
  tweet_text = str(tweet).lower()
  if "food" in tweet_text or "hungry" in tweet_text or "meal" in tweet_text or "snacks" in tweet_text or "eat" in tweet_text:
    return 'Food'
  elif "medical" in tweet_text or "medicine" in tweet_text or "injury" in tweet_text or "hospital" in tweet_text:
    return 'Medical'
  elif "energy" in tweet_text or "power" in tweet_text or "electricity" in tweet_text:
    return 'Energy'
  elif "water" in tweet_text or "thirsty" in tweet_text or "drink" in tweet_text:
    return 'Water'
  else:
    return 'None'



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

  classes = sorted(class_counts.keys())

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

  # for c in classes:
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
  for (f,(e,prob_classes_given_feature)) in feat2entropy[:10]:
    print "%s: entropy %.2f, " % (f,e),
    for c_idx,c in enumerate(sorted(class_counts.keys())):
      print "%s: %.4f, " % (c, prob_classes_given_feature[c_idx]),
    print ""
  print "\nWords with lowest cross entropy:"
  for (f,(e,prob_classes_given_feature)) in feat2entropy[-10:]:
    print "%s: entropy %.2f, " % (f,e),
    for c_idx,c in enumerate(sorted(class_counts.keys())):
      print "%s: %.4f, " % (c, prob_classes_given_feature[c_idx]),
    print ""

  feat2p_c_given_f = {k:v[1] for (k,v) in feat2entropy}
  for c_idx,c in enumerate(classes):
    c_prob = class_probs[c]
    feat2p_c_given_f_lst = [(f, feat2p_c_given_f[f][c_idx]) for f in all_features]
    feat2p_c_given_f_lst = sorted(feat2p_c_given_f_lst, key=lambda x: x[1], reverse=True)
    print "Top features for %s which has base prob %.4f" % (c, c_prob)
    for (f, p_c_given_f) in feat2p_c_given_f_lst[:10]:
      print "%s: %.4f (%i/%i occurrences)" % (f, p_c_given_f, feature_counts[c][f], feature_only_counts[f]) # features that maximize p(C|f)
    print ""


  return class_probs, feature_probs

def evaluate(tweet2prediction, classes):
  # get unbalanced accuracy
  num_correct = len([1 for tweet,predicted_label in tweet2prediction.iteritems() if tweet.category==predicted_label])
  total = len(tweet2prediction.keys())
  acc = float(num_correct)*100/float(total)
  print "%i correct of %i. Unbalanced accuracy = %.2f percent\n" % (num_correct, total, acc)

  # get recall / precision / F1 per class
  class2f1 = {}
  for c in classes:
    num_correct = len([1 for tweet,predicted_label in tweet2prediction.iteritems() if tweet.category==c and predicted_label==c])

    num_fp = len([1 for tweet,predicted_label in tweet2prediction.iteritems() if tweet.category!=c and predicted_label==c])

    num_fn = len([1 for tweet,predicted_label in tweet2prediction.iteritems() if tweet.category==c and predicted_label!=c])

    precision = float(num_correct)*100 / float(num_correct + num_fp)
    recall = float(num_correct)*100 / float(num_correct + num_fn)

    f1 = 2*precision*recall / (precision+recall)

    print "Class %s: precision %.2f, recall %.2f, F1 %.2f" % (c, precision, recall, f1)
    class2f1[c]=f1
  balanced_f1 = sum(class2f1.values())/len(class2f1)
  print "Balanced F1: %.2f percent\n" % (balanced_f1)


def get_conf_mat(tweet2prediction, classes):
  """prints confusion matrix. rows are gold label, columns are predicted label."""
  num_classes = len(classes)
  conf_mat = np.zeros((num_classes, num_classes), dtype=np.int32)
  for tweet,predicted_label in tweet2prediction.iteritems():
    try:
      gold_idx = classes.index(tweet.category)
    except:
      print tweet
      print tweet.category
      print classes
      exit()
    predicted_idx = classes.index(predicted_label)
    conf_mat[gold_idx, predicted_idx] += 1
  print "Confusion matrix classes: ", classes
  print conf_mat


def classify_unseen_nb(test_tweets, class_probs, feature_probs, categories):
  tweet2prediction = {tweet: classify_nb(tweet, class_probs, feature_probs) for tweet in test_tweets} # maps each unseen tweet to its predicted label
  print "\nEvaluating NB system on test set..."
  evaluate(tweet2prediction, categories)
  get_conf_mat(tweet2prediction, categories)


def classify_unseen_rb(test_tweets, categories):
  tweet2prediction = {tweet: classify_rb(tweet) for tweet in test_tweets} # maps each unseen tweet to its predicted label
  print "\nEvaluating rule-based system on test set..."
  evaluate(tweet2prediction, categories)
  get_conf_mat(tweet2prediction, categories)


if __name__ == "__main__":
  train_tweets, test_tweets, categories, needs = read_data()
  class_probs, feature_probs = learn_nb(train_tweets)
  classify_unseen_nb(test_tweets, class_probs, feature_probs, categories)
  classify_unseen_rb(test_tweets, categories)

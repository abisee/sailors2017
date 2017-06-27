#!/usr/bin/env python

import lib

def main():
    data = lib.read_csv('data/labeled-data-singlelabels.csv')
    train_tweets, test_tweets = lib.split_data(data)

    lib.write_csv(train_tweets, 'data/labeled-data-singlelabels-train.csv')
    lib.write_csv(test_tweets, 'data/labeled-data-singlelabels-test.csv')


    train_tweets2, test_tweets2 = lib.read_data()

    assert(len(train_tweets) == len(train_tweets2))
    assert(len(test_tweets) == len(test_tweets2))



if __name__ == '__main__':
    main()

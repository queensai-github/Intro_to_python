import os

import nltk


class TextProcessor(object):

    def __init__(self):
        # Check if the stopwords corpus is downloaded, else download it
        if not os.path.exists(os.path.join(os.environ['HOME'], 'nltk_data', 'corpora', 'stopwords')):
            nltk.download('stopwords')

        from nltk.corpus import stopwords
        self.stop_words = stopwords.words('english')

    def remove_mentions(self, text):
        return text

    def unescape_html(self, text):
        return text

    def remove_punctuation(self, text, keep_emoticons=True):
        return text

    def lower(self, text):
        return text

    def remove_stopwords(self, text):
        return text

    def stem(self, text):
        return text

    def remove_extra_spaces(self, text):
        return text

    def preprocess(self, df):
        """ Runs all pre-processing steps in a dataframe of tweets. """

        print('Preprocessing data...')
        df['text'] = df.text.apply(self.remove_mentions) \
                            .apply(self.unescape_html) \
                            .apply(self.remove_punctuation) \
                            .apply(self.lower) \
                            .apply(self.remove_stopwords) \
                            .apply(self.stem) \
                            .apply(self.remove_extra_spaces)
        return df

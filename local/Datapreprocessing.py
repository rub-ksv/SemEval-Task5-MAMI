# -*- coding: utf-8 -*-
import json
import os, sys
import html
import nltk
import random
import re
from nltk.tokenize import TweetTokenizer
from ftfy import fix_text
from emoji import UNICODE_EMOJI
from emoji import demojize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

import unicodedata
import gensim
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
tokenizer = TweetTokenizer()
SEED=1734176512
random.seed(SEED)
nltk.download('words')
words = set(nltk.corpus.words.words())

def normalizeTFToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return token.replace('@', '')
    elif token.startswith("#"):
        return token.replace('#', '')
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return 'URL'
    elif token in UNICODE_EMOJI['en']:
        return demojize(token)
    else:
        return token

def nospecial(text):
	import re
	text = re.sub("[^a-zA-Z0-9 .,?!\']+", "",text)
	return text


def main(Datadir, Savedir, ifdebug, num):
    tokenizer = TweetTokenizer()
    if not os.path.exists(Savedir):
        os.makedirs(Savedir)

    if ifdebug == 'true':
        dsets = ['training']
    else:
        dsets = ['training', 'test']
    for dset in dsets:
        setsdir = os.path.join(Datadir, dset, dset + '.csv')
        imagemaindir = os.path.join(Datadir, dset)
        with open(setsdir, encoding="utf8") as f:
            data = f.readlines()
        with open(os.path.join(Savedir, 'imagecap', dset, dset + "_image_to_text.json"), encoding="utf8") as json_file:
            train_features = json.load(json_file)
        del data[0]

        if dset == 'training':
            datadict = {}
            for i in data:
                split = i.split('\t')
                imagedir = os.path.join(imagemaindir, split[0])
                misogynous = split[1]
                shaming = split[2]
                stereotype = split[3]
                objectification = split[4]
                violence = split[5]
                texttrans = split[6].strip('\n')
                texttrans = fix_text(texttrans)  # some special chars like 'à¶´à¶§à·’ à¶»à·à¶½à·Š'
                # will transformed into the right form පටි රෝල්

                tokens = tokenizer.tokenize(texttrans.replace('\n', ''))
                normTweet = " ".join(filter(None, [normalizeTFToken(token) for token in tokens]))

                normTweet = normTweet.replace(' ’ ', '’')
                normTweet = normTweet.replace(' .', '.')
                normTweet = normTweet.replace(' ,', ',')
                normTweet = normTweet.replace(' ?', '?')
                normTweet = normTweet.replace(' !', '!')

                texttrans = nospecial(normTweet)
                # texttrans = tool.correct(texttrans)
                texttrans = texttrans.lower()

                imagetext = list(train_features[split[0]].values())[0]
                imagetext = imagetext.replace('a picture of a blurry image of ', '')
                imagetext = imagetext.replace('a picture of a blurry photo of ', '')
                imagetext = imagetext.replace('a blurry image of ', '')
                imagetext = imagetext.replace('a blurry picture of ', '')
                imagetext = imagetext.replace('a blurry photo of ', '')
                imagetext = imagetext.replace('a blurry image and ', '')
                imagetext = imagetext.replace('a blurry picture and ', '')
                imagetext = imagetext.replace('a blurry photo and ', '')

                texttrans = texttrans + '. ' + imagetext
                fileid = split[0] + '_' + dset
                datadict.update({fileid: {}})
                datadict[fileid].update({'image': imagedir})
                datadict[fileid].update({'misogynous': misogynous})
                datadict[fileid].update({'shaming': shaming})
                datadict[fileid].update({'stereotype': stereotype})
                datadict[fileid].update({'objectification': objectification})
                datadict[fileid].update({'violence': violence})
                datadict[fileid].update({'text': texttrans})
                datadict[fileid].update({'imagetext': imagetext})



            if ifdebug == 'true':
                datakeys = list(datadict.keys())
                testkeys = random.sample(datakeys, int(num))
                traindict = {}
                testdict = {}
                for key in datakeys:
                    if key in testkeys:
                        testdict.update({key: datadict[key]})
                    else:
                        traindict.update({key: datadict[key]})
                with open(os.path.join(Savedir, "training.json"), 'w', encoding='utf-8') as f:
                    json.dump(traindict, f, ensure_ascii=False, indent=4)
                with open(os.path.join(Savedir, "test.json"), 'w', encoding='utf-8') as f:
                    json.dump(testdict, f, ensure_ascii=False, indent=4)
            else:
                with open(os.path.join(Savedir, dset + ".json"), 'w', encoding='utf-8') as f:
                    json.dump(datadict, f, ensure_ascii=False, indent=4)
        else:
            datadict = {}
            for i in data:
                split = i.split('\t')
                imagedir = os.path.join(imagemaindir, split[0])
                texttrans = split[1].strip('\n')
                texttrans = fix_text(texttrans)  # some special chars like 'à¶´à¶§à·’ à¶»à·à¶½à·Š'
                # will transformed into the right form පටි රෝල්

                tokens = tokenizer.tokenize(texttrans.replace('\n', ''))
                normTweet = " ".join(filter(None, [normalizeTFToken(token) for token in tokens]))

                normTweet = normTweet.replace(' ’ ', '’')
                normTweet = normTweet.replace(' .', '.')
                normTweet = normTweet.replace(' ,', ',')
                normTweet = normTweet.replace(' ?', '?')
                normTweet = normTweet.replace(' !', '!')

                texttrans = nospecial(normTweet)
                # texttrans = tool.correct(texttrans)
                texttrans = texttrans.lower()

                imagetext = list(train_features[split[0]].values())[0]
                imagetext = imagetext.replace('a picture of a blurry image of ', '')
                imagetext = imagetext.replace('a picture of a blurry photo of ', '')
                imagetext = imagetext.replace('a blurry image of ', '')
                imagetext = imagetext.replace('a blurry picture of ', '')
                imagetext = imagetext.replace('a blurry photo of ', '')
                imagetext = imagetext.replace('a blurry image and ', '')
                imagetext = imagetext.replace('a blurry picture and ', '')
                imagetext = imagetext.replace('a blurry photo and ', '')
                texttrans = texttrans + '. ' + imagetext
                fileid = split[0] + '_' + dset
                datadict.update({fileid: {}})
                datadict[fileid].update({'image': imagedir})


                datadict[fileid].update({'text': texttrans})
                datadict[fileid].update({'imagetext': imagetext})

            if ifdebug == 'true':
                datakeys = list(datadict.keys())
                testkeys = random.sample(datakeys, int(num))
                traindict = {}
                testdict = {}
                for key in datakeys:
                    if key in testkeys:
                        testdict.update({key: datadict[key]})
                    else:
                        traindict.update({key: datadict[key]})
                with open(os.path.join(Savedir, "training.json"), 'w', encoding='utf-8') as f:
                    json.dump(traindict, f, ensure_ascii=False, indent=4)
                with open(os.path.join(Savedir, "test.json"), 'w', encoding='utf-8') as f:
                    json.dump(testdict, f, ensure_ascii=False, indent=4)
            else:
                with open(os.path.join(Savedir, dset + ".json"), 'w', encoding='utf-8') as f:
                    json.dump(datadict, f, ensure_ascii=False, indent=4)

# hand over parameter overview
# sys.argv[1] = Datadir (str): Dataset dir
# sys.argv[2] = Savedir (str): Where save processing daata
# sys.argv[3] = ifdebug (str): If ifdebug is 'true', split num(next variable) Samples from training set as intern test set
# sys.argv[4] = num (str): Split how many sample from training set if ifdebug is 'true'
main(sys.argv[1],sys.argv[2],sys.argv[3], sys.argv[4])

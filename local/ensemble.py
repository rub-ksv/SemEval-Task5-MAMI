
import os, sys
import io
import copy
import numpy as np
import torch
import pandas as pd

from shutil import copyfile
import glob
import pickle
from collections import OrderedDict
from sklearn import metrics
from evaluation import *


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def load_data(path, nsplit):
    outdict = {}
    for n in range(nsplit):
        filedir = os.path.join(path, str(n), 'output.pkl')
        with open(filedir, 'rb') as f:
            out = CPU_Unpickler(f).load()
        out = OrderedDict(sorted(out.items()))
        outdict.update({n: out})

    return outdict

def hard_vote(models):
    outkeys = list(models[0].keys())
    filename = []
    labelsdim = list(models[0][outkeys[0]]['logprob'].size())
    if labelsdim[0] == 1:
        label = np.zeros((1000, len(models)))
        for keyid in range(len(outkeys)):
            filename.append(outkeys[keyid])
            for modelid in range(len(models)):
                bound = 0.8
                '''if modelid == 2 or modelid == 3:
                    bound = 0.7
                else:
                    bound = 0.8'''
                if models[modelid][outkeys[keyid]]['logprob'] > bound:
                    label[keyid][modelid] = 1
                else:
                    label[keyid][modelid] = 0
        label = np.expand_dims((np.sum(label, axis=-1) > np.floor(len(models) / 2)).astype('int'), axis=-1)
    else:
        label1 = np.zeros((1000, len(models)))
        label2 = np.zeros((1000, len(models)))
        label3 = np.zeros((1000, len(models)))
        label4 = np.zeros((1000, len(models)))
        for keyid in range(len(outkeys)):
            filename.append(outkeys[keyid])
            for modelid in range(len(models)):
                label1[keyid][modelid] = models[modelid][outkeys[keyid]]['predict'][0]
                label2[keyid][modelid] = models[modelid][outkeys[keyid]]['predict'][1]
                label3[keyid][modelid] = models[modelid][outkeys[keyid]]['predict'][2]
                label4[keyid][modelid] = models[modelid][outkeys[keyid]]['predict'][3]
        if (len(models) % 2) == 0:
            label1 = np.expand_dims((np.sum(label1, axis=-1) >= np.floor(len(models) / 2)).astype('int'), axis=-1)
            label2 = np.expand_dims((np.sum(label2, axis=-1) >= np.floor(len(models) / 2)).astype('int'), axis=-1)
            label3 = np.expand_dims((np.sum(label3, axis=-1) >= np.floor(len(models) / 2)).astype('int'), axis=-1)
            label4 = np.expand_dims((np.sum(label4, axis=-1) >= np.floor(len(models) / 2)).astype('int'), axis=-1)
        else:
            label1 = np.expand_dims((np.sum(label1, axis=-1) > np.floor(len(models) / 2)).astype('int'), axis=-1)
            label2 = np.expand_dims((np.sum(label2, axis=-1) > np.floor(len(models) / 2)).astype('int'), axis=-1)
            label3 = np.expand_dims((np.sum(label3, axis=-1) > np.floor(len(models) / 2)).astype('int'), axis=-1)
            label4 = np.expand_dims((np.sum(label4, axis=-1) > np.floor(len(models) / 2)).astype('int'), axis=-1)

        '''label1 = np.expand_dims((np.sum(label1, axis=-1) > np.floor(2)).astype('int'), axis=-1)
        label2 = np.expand_dims((np.sum(label2, axis=-1) > np.floor(2)).astype('int'), axis=-1)
        label3 = np.expand_dims((np.sum(label3, axis=-1) > np.floor(2)).astype('int'), axis=-1)
        label4 = np.expand_dims((np.sum(label4, axis=-1) > np.floor(2)).astype('int'), axis=-1)'''
        label = np.concatenate((label1, label2, label3, label4), axis=-1)


    return label, filename

def soft_vote(models, weights):
    weights = list(weights.values())
    weights = [x / sum(weights) for x in weights]
    outdict = copy.deepcopy(models[0])
    outkeys = list(models[0].keys())
    outkeys.sort()
    for outkey in outkeys:
        prob = []
        for modelid in range(len(models)):
            prob.append(weights[modelid] * models[modelid][outkey]['logprob'])
        prob = sum(prob)
        outdict[outkey]['logprob'] = prob
        outdict[outkey]['predict'] = torch.round(prob)



    return outdict

def soft_vote_model(models, weightdict):
    modelsout = {}
    weightmodel = {}
    modelkeys = weightdict.keys()
    for modelid in modelkeys:
        newmodel = soft_vote(models[modelid], weightdict[modelid])
        modelsout.update({modelid: newmodel})
        maxscore = max(list(weightdict[modelid].values()))
        weightmodel.update({modelid: maxscore})
    return modelsout, weightmodel


def hard_vote_model(models, ifpart=False):
    if ifpart == False:
        label, filename = hard_vote(models)
    else:
        ## text model fusion
        labeltext, filename = hard_vote(models[:3])
        textmodel = models[0]
        for i in range(len(filename)):
            textmodel[filename[i]]['predict'] = torch.FloatTensor(labeltext[i])
        label, filename = hard_vote([textmodel, models[-2], models[-1]])


    return label, filename

def single_model(models):
    outkeys = list(models[0].keys())
    filename = []
    label = []
    for keyid in range(len(outkeys)):
        filename.append(outkeys[keyid])
        label.append(models[0][outkeys[keyid]]['predict'].unsqueeze(0))

    label = np.concatenate(label, axis=0)

    return label, filename

def soft_vote_feat_model(models, weightdict):
    modellist = list(models.keys())
    weights = list(weightdict.values())
    weights = [x / sum(weights) for x in weights]
    modelsout = {}
    weightmodel = {}
    outdict = copy.deepcopy(models['BERTC'])
    outkeys = list(models['BERTC'].keys())
    outkeys.sort()
    for outkey in outkeys:
        prob = []
        for modelid in range(len(models)):
            prob.append(weights[modelid] * models[modellist[modelid]][outkey]['logprob'])
        prob = sum(prob)
        outdict[outkey]['logprob'] = prob
        outdict[outkey]['predict'] = torch.round(prob)
    label, filename = single_model([outdict])

    return label, filename

def load_weight(filedir, models, nsplit):
    text_file = open(filedir, "r")
    lines = text_file.readlines()
    for i in range(10):
        lines.append('\n')
    weightdict = {}
    for model in models:
        weightdict.update({model: {}})
        for n in range(nsplit):
            for i in range(len(lines)):
                if model + '/' + str(n) in lines[i]:
                    epochscore = []
                    nextid = 1
                    while lines[i + nextid] != '\n':
                        if 'modelavg2' in lines[i + nextid]:
                            pass
                        else:
                            text = lines[i + nextid].split('_')
                            evalf1 = float(text[-1].strip('.pkl\n'))
                            epochscore.append(evalf1)

                        nextid = nextid + 1
                    bestscore = max(epochscore)
                    weightdict[model].update({n: bestscore})
                    #weightdict.update({model + '_' + str(n): bestscore})
    return weightdict


def ensemble(savedir, refdir, logdir):
    print(savedir)
    print(refdir)
    print(logdir)
    bertcdir = os.path.join(savedir, 'classification',
                            'ocr', 'BERTC')
    gcandir = os.path.join(savedir, 'classification',
                           'ocr', 'GCAN')
    imagedir = os.path.join(savedir, 'classification',
                            'image')
    bidir = os.path.join(savedir, 'classification',
                         'bi')                          ## using 3 streams, BERTC, GCAN and image
    bibertcdir = os.path.join(savedir, 'classification',
                               'BERTC_bi')              ## using 2 streams, BERTC and image
    bigcandir = os.path.join(savedir, 'classification',
                               'GCAN_bi')               ## using 2 stream, GCAN and image
    bibertcgcandir = os.path.join(savedir, 'classification',
                             'bi_BERTC_GCAN')           ## using 2 stream, BERTC and GCAN

    bertcresults = load_data(bertcdir, 10)
    gcanresults = load_data(gcandir, 10)
    imageresults = load_data(imagedir, 10)
    biresults = load_data(bidir, 10)
    bertcbiresults = load_data(bibertcdir, 10)
    gcanbiresults = load_data(bigcandir, 10)
    bibertcgcanresults = load_data(bibertcgcandir, 10)

    modeldict = {}
    modeldict.update({'BERTC': bertcresults})
    modeldict.update({'GCAN': gcanresults})
    modeldict.update({'image': imageresults})
    modeldict.update({'bi': biresults})
    modeldict.update({'BERTC_bi': bertcbiresults})
    modeldict.update({'GCAN_bi': gcanbiresults})
    modeldict.update({'bi_BERTC_GCAN': bibertcgcanresults})
    modellists = [['image', 'bi', 'GCAN_bi', 'BERTC_bi'],
                ['BERTC_bi', 'bi']]
    for listi in range(len(modellists)):
        modellist = modellists[listi]
        weightdict = load_weight(logdir, modellist, 10)

        soft_voted_models, weightdict = soft_vote_model(modeldict, weightdict)
        # label, filename = soft_vote_model([bertcresults, gcanresults, gloveresults, imageresults, biresults],
        #                            [0.7421, 0.7488, 0.6478, 0.6681, 0.7587])
        # del soft_voted_models['image']
        label, filename = hard_vote_model(list(soft_voted_models.values()), ifpart=False)
        # label, filename = soft_vote_feat_model(soft_voted_models, weightdict)

        nohate = 0
        shaming = 0
        stero = 0
        object = 0
        voi = 0
        prediction_out = np.zeros((1000, 5))
        predictions = label
        for j in range(len(predictions)):
            prediction_out[j, 1:] = predictions[j]
            if sum(predictions[j]) == 0:
                prediction_out[j][0] = 0
            else:
                prediction_out[j][0] = 1
            if prediction_out[j][0] == 0:
                nohate = nohate + 1
            if prediction_out[j][1] == 1:
                shaming = shaming + 1
            if prediction_out[j][2] == 1:
                stero = stero + 1
            if prediction_out[j][3] == 1:
                object = object + 1
            if prediction_out[j][4] == 1:
                voi = voi + 1
        print('non-hate: ' + str(nohate))
        print('shaming: ' + str(shaming))
        print('stero: ' + str(stero))
        print('object: ' + str(object))
        print('voilence: ' + str(voi))

        predictions_db = pd.DataFrame(prediction_out,
                                      columns=['misogynous', 'shaming', 'stereotype', 'objectification',
                                               'violence'])
        predictions_db = predictions_db.apply(lambda x: list(map(int, x)))
        predictions_db['file_name'] = filename

        predictions_db = predictions_db[
            ['file_name', 'misogynous', 'shaming', 'stereotype', 'objectification', 'violence']]

        saveoutdir = os.path.join(savedir, 'answer' + str(listi) + '.txt')
        predictions_db.to_csv(saveoutdir, index=False, sep='\t', header=False)
        main(saveoutdir, refdir, savedir, listi)
        a = 1



'''savedir = './../results'
refdir = './../MAMI/test_labels.txt'
logdir = './../classification.txt'
ensemble(savedir, refdir, logdir)'''
# hand over parameter overview
# sys.argv[1] = savedir (str): Results dir
# sys.argv[2] = refdir (str): where test set ground turth saved
# sys.argv[3] = logdir (str): List all trained model information
ensemble(sys.argv[1], sys.argv[2], sys.argv[3])














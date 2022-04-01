import pickle
import torch
import numpy as np
from torchvision import transforms
import os, sys, json
import collections
import random
from models import *
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import trange
from sklearn.model_selection import KFold
from utils import *
from transformers import RobertaTokenizer, RobertaModel
SEED=1734176512
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True


class Multilabelsloss(torch.nn.Module):
    def __init__(self, weight):
        super(Multilabelsloss, self).__init__()
        self.BCE = torch.nn.BCELoss()
        self.MSE = torch.nn.MSELoss()
        self.weight = weight

    def forward(self, logits, targets):
        '''
        Compute loss: kl - evi

        '''
        firsttarget = targets[:, 0]
        maxkateg, _ = torch.max(logits, dim=-1)
        mseloss = self.MSE(maxkateg, firsttarget)


        scoremis = 10000 / 5000  # N / N_1
        scoresham = 10000 / 1274
        scorestere = 10000 / 2810
        scoreobj = 10000 / 2202
        scorevio = 10000 / 953
        summer = (scoresham + scorestere + scoreobj + scorevio)
        scoresham = scoresham / summer
        scorestere = scorestere / summer
        scoreobj = scoreobj / summer
        scorevio = scorevio / summer
        losssham = self.BCE(logits[:, 0], targets[:, 1])
        lossstere = self.BCE(logits[:, 1], targets[:, 2])
        lossobj = self.BCE(logits[:, 2], targets[:, 3])
        lossvio = self.BCE(logits[:, 3], targets[:, 4])
        multilabelloss = (scoresham * losssham + scorestere * lossstere + scoreobj * lossobj + scorevio * lossvio)
        loss = multilabelloss + self.weight * mseloss

        return loss



'''if __name__ == '__main__':
    Datadir = './../Dataset'
    tasktype = 'classification'
    featype = 'bert'
    mainSavedir = "./../model/"
    Resultsdir = './../results/'
    adjdir = './../'
    stream = 'ocr'
    best_model = 'acc'
    ifgpu = 'false'
    iffinetune = 'false'
    start = '0'''
def train(Datadir, tasktype, featype, mainSavedir, Resultsdir, stream, best_model, ifgpu, iffinetune, start, adjdir):
    Savedir = os.path.join(mainSavedir, tasktype, stream)
    Resultsdir = os.path.join(Resultsdir, tasktype, stream)
    if ifgpu == 'true':
        device = 'cuda'
    else:
        device = 'cpu'
    if not os.path.exists(Savedir):
        os.makedirs(Savedir)

    if not os.path.exists(Resultsdir):
        os.makedirs(Resultsdir)

    ## Configuration
    N_AVERAGE = 2 # average best 2 models
    TRAIN_BATCH_SIZE = 2#16#8
    DEV_BATCH_SIZE = 2#16#8
    EVAL_BATCH_SIZE = 1#8
    FINE_TUNING_TRAIN_BATCH_SIZE = 32#4
    FINE_TUNING_DEV_BATCH_SIZE = 32#4
    FINE_TUNING_EVAL_BATCH_SIZE = 1#4
    NUM_TRAIN_EPOCHS = 100
    NWORKER = 16
    nsplit = 10
    lr = 1e-5 #2e-5
    finetunelr = 5e-6# 5e-6
    p = 10


    ########## conf transformer file
    conf = {}
    conf.update({'adim': 1024})
    conf.update({'transformer-attn-dropout-rate': 0.0})
    conf.update({'nheads': 8})
    conf.update({'elayers': 2})
    conf.update({'transformer-input-layer': 'linear'})
    conf.update({'dropout-rate': 0.5})
    conf.update({'padding_index': 1})

    ## load train features
    with open(os.path.join(Datadir, "training.json"), encoding="utf8") as json_file:
        train_features = json.load(json_file)

    ## load test features
    with open(os.path.join(Datadir, "test.json"), encoding="utf8") as json_file:
        test_features = json.load(json_file)

    '''dict_items = train_features.items()
    train_featureslist = list(dict_items)[:80]
    train_features = {}
    for i in train_featureslist:
        train_features.update({i[0]: i[1]})

    dict_items = test_features.items()
    test_featureslist = list(dict_items)[:80]
    test_features = {}
    for i in test_featureslist:
        test_features.update({i[0]: i[1]})'''

    transform = transforms.Compose(
        [
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor()
        ]
    )

    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    dataset = TextLevelGNNDatasetClass(train_file=train_features,
                                       test_file=test_features,
                                       tokenizer=tokenizer,
                                       tasktype=tasktype,
                                       p=p,
                                       min_freq=2,
                                       stream=stream,
                                       transform=transform,
                                       adjdir=adjdir)
    train_examples_len = len(dataset.train_dataset)
    print(train_examples_len)
    kfold = KFold(n_splits=nsplit, random_state=1, shuffle=True)

    if stream == 'ocr':
        # Start print
        print('--------------------------------')
        for modeltype in ['BERTC', 'GCAN']:
            for fold, (train_ids, dev_ids) in enumerate(kfold.split(dataset.train_dataset)):
                if fold < int(start):
                    pass
                else:
                    modelResultsdir = os.path.join(Resultsdir, modeltype)
                    if not os.path.exists(modelResultsdir):
                        os.makedirs(modelResultsdir)
                    modelSavedir = os.path.join(Savedir, modeltype)
                    if not os.path.exists(modelSavedir):
                        os.makedirs(modelSavedir)

                    # Print
                    print(f'FOLD {fold}')
                    print('--------------------------------')
                    evalacc_best = 0
                    evalloss_best = np.Inf
                    early_wait = 4
                    run_wait = 1
                    continuescore = 0
                    stop_counter = 0
                    ## make model
                    if modeltype == 'BERTC':
                        model = E2EBERTC(1024, 4, conf, device)
                    elif modeltype == 'GCAN':
                        model = E2EGCAN(1024, 4, conf, device)
                    model = model.to(device)

                    optimizer = AdamW(model.parameters(),
                                          lr=lr,
                                          eps=1e-8, weight_decay=0.01
                                          )
                    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                                    num_warmup_steps=int(
                                                                        0.9 * train_examples_len / TRAIN_BATCH_SIZE) * 4,
                                                                    num_training_steps=int(
                                                                        0.9 * train_examples_len / TRAIN_BATCH_SIZE) * 50)


                    Resultsnewdir = os.path.join(modelResultsdir, str(fold))
                    if not os.path.exists(Resultsnewdir):
                        os.makedirs(Resultsnewdir)
                    modelnewdir = os.path.join(modelSavedir, str(fold))
                    if not os.path.exists(modelnewdir):
                        os.makedirs(modelnewdir)
                    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
                    dev_subsampler = torch.utils.data.SubsetRandomSampler(dev_ids)
                    dev_examples_len = len(dev_subsampler)
                    print(dev_examples_len)

                    data_loader_train = torch.utils.data.DataLoader(dataset.train_dataset,
                                                                    batch_size=TRAIN_BATCH_SIZE,
                                                                    num_workers=NWORKER, drop_last=True,
                                                                    sampler=train_subsampler,
                                                                    collate_fn=pad_custom_sequence)

                    data_loader_dev = torch.utils.data.DataLoader(dataset.train_dataset, batch_size=DEV_BATCH_SIZE,
                                                                  num_workers=NWORKER, drop_last=True,
                                                                  sampler=dev_subsampler,
                                                                  collate_fn=pad_custom_sequence)

                    criterion = Multilabelsloss(0.1)
                    for epoch in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
                        if modeltype == 'BERTC' or modeltype == 'GCAN':
                            if iffinetune == 'false':
                                for param in model.BERTtext.parameters():
                                    # for param in model.module.BERT.parameters():
                                    param.requires_grad = False
                            else:
                                for param in model.BERTtext.parameters():
                                    # for param in model.module.BERT.parameters():
                                    param.requires_grad = True

                        torch.cuda.empty_cache()
                        tr_loss = 0
                        nb_tr_examples, nb_tr_steps = 0, 0
                        model.train()
                        trainpredict = []
                        trainlabel = []
                        for count, batch in enumerate(data_loader_train, 0):
                            node_sets = batch[0].to(device)
                            masks = batch[1]
                            word_weight = batch[2].to(device)
                            label = batch[3].to(device)

                            mask = torch.sum(masks, dim=-1)
                            mask = mask  # - 2 # we remove both <s> and </s> symbol
                            mask_onehot = creat_mask(mask)
                            masks = masks.to(device)
                            mask_onehot = mask_onehot.to(device)
                            mask = mask.to(device)


                            embfeat, cls = model.textbertmodel(node_sets, masks)
                            logits, _ = model.forward(embfeat, cls, word_weight, mask_onehot, mask)

                            loss = criterion(logits, label)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                            print("\r%f" % loss, end='')

                            tr_loss += loss.item()
                            nb_tr_steps += 1
                            trainpredict.extend(torch.round(logits.cpu().detach()).tolist())
                            trainlabel.extend(label.cpu().data.numpy().tolist())

                        results = []
                        total_occurences = 0
                        for index in range(1, 5):
                            label = []
                            predict = []
                            for i in range(len(trainlabel)):
                                label.extend([trainlabel[i][index]])
                                if index == 0:
                                    predict.extend([int(sum(trainpredict[i]) > 0)])
                                else:
                                    predict.extend([trainpredict[i][index - 1]])
                            f1_score = compute_f1(predict, label)
                            f1weight = label.count(True)
                            total_occurences += f1weight
                            results.append(f1_score * f1weight)
                        trainallscore = sum(results) / total_occurences
                        ## Early Stopping
                        print('Early Stoppong check')
                        torch.cuda.empty_cache()
                        evallossvec = []
                        evalacc = 0
                        model.eval()
                        evalpredict = []
                        evalresults = []
                        evallabel = []
                        for dev_step, dev_batch in enumerate(data_loader_dev):
                            node_sets = dev_batch[0].to(device)
                            masks = dev_batch[1]
                            word_weight = dev_batch[2].to(device)
                            label = dev_batch[3].to(device)

                            mask = torch.sum(masks, dim=-1)
                            mask = mask  # - 2  # we remove both <s> and </s> symbol
                            mask_onehot = creat_mask(mask)
                            masks = masks.to(device)
                            mask_onehot = mask_onehot.to(device)
                            mask = mask.to(device)
                            with torch.no_grad():
                                embfeat, cls = model.textbertmodel(node_sets, masks)
                                dev_logits, _ = model.forward(embfeat, cls, word_weight, mask_onehot, mask)
                            evalresults.append(dev_logits.cpu().data.numpy())
                            dev_loss = criterion(dev_logits, label)
                            evalpredict.extend(torch.round(dev_logits.cpu().detach()).tolist())
                            evallabel.extend(label.cpu().data.numpy().tolist())
                            evallossvec.append(dev_loss.cpu().data.numpy())
                        results = []
                        total_occurences = 0
                        for index in range(1, 5):
                            label = []
                            predict = []
                            for i in range(len(evallabel)):
                                label.extend([evallabel[i][index]])
                                if index == 0:
                                    predict.extend([int(sum(evalpredict[i]) > 0)])
                                else:
                                    predict.extend([evalpredict[i][index - 1]])
                            f1_score = compute_f1(predict, label)
                            f1weight = label.count(True)
                            total_occurences += f1weight
                            results.append(f1_score * f1weight)
                        allscore = sum(results) / total_occurences
                        evalresults = (np.array(evalresults)).reshape(-1, 1)
                        evallabel = (np.array(evallabel)).reshape(-1, 1)
                        np.savetxt(os.path.join(Resultsnewdir, 'results' + str(epoch) + '.txt'), evalresults,
                                   fmt='%.3f',
                                   delimiter=' ',
                                   newline='\n', header='', footer='', comments='# ',
                                   encoding=None)
                        np.savetxt(os.path.join(Resultsnewdir, 'label' + str(epoch) + '.txt'), evallabel,
                                   fmt='%.3f',
                                   delimiter=' ',
                                   newline='\n', header='', footer='', comments='# ',
                                   encoding=None)

                        evallossmean = np.mean(np.array(evallossvec))
                        for param_group in optimizer.param_groups:
                            currentlr = param_group['lr']
                        OUTPUT_DIR = os.path.join(modelnewdir,
                                                  str(epoch) + '_' + str(evallossmean) + '_' + str(
                                                      currentlr) + '_' + str(trainallscore)[:6] + '_' + str(
                                                      allscore)[:6] + '.pkl')
                        if not os.path.exists(Savedir):
                            os.makedirs(Savedir)
                        torch.save(model, OUTPUT_DIR)

                        torch.cuda.empty_cache()
                        if allscore < evalacc_best:
                            stop_counter = stop_counter + 1
                            print('no improvement')
                            continuescore = 0
                        else:
                            print('new score')
                            evalacc_best = allscore
                            continuescore = continuescore + 1

                        if continuescore >= run_wait:
                            stop_counter = 0
                        print(stop_counter)
                        print(early_wait)
                        if stop_counter < early_wait:
                            pass
                        else:
                            break

                    netlist = os.listdir(os.path.join(modelnewdir))
                    netlist.sort()
                    net_dict = {}
                    if best_model == 'loss':
                        for m in range(len(netlist)):
                            templist = netlist[m].split('_')
                            net_dict[templist[1]] = netlist[m]
                        net_dict = collections.OrderedDict(sorted(net_dict.items(), reverse=True))
                    else:
                        for m in range(len(netlist)):
                            templist = netlist[m].split('_')
                            net_dict[templist[4]] = netlist[m]
                        net_dict = collections.OrderedDict(sorted(net_dict.items()))

                    if N_AVERAGE >= 2:
                        avg = None
                        for n in range(N_AVERAGE):
                            netname = net_dict.get(list(net_dict)[-(n + 1)])
                            print(netname)
                            net = torch.load(os.path.join(modelnewdir, netname)).state_dict()
                            if avg is None:
                                avg = net
                            else:
                                for k in avg.keys():
                                    avg[k] += net[k]
                        for k in avg.keys():
                            if avg[k] is not None:
                                if avg[k].is_floating_point():
                                    avg[k] /= 2
                                else:
                                    avg[k] //= 2
                        model.load_state_dict(avg)
                        torch.save(model, (os.path.join(modelnewdir, 'modelavg2.pkl')))
                    else:
                        netname = net_dict.get(list(net_dict)[-1])

                    model = torch.load(os.path.join(modelnewdir, 'modelavg2.pkl'))
                    data_loader_eval = torch.utils.data.DataLoader(dataset.test_dataset, batch_size=EVAL_BATCH_SIZE,
                                                                   shuffle=False,
                                                                   drop_last=False, num_workers=NWORKER,
                                                                   collate_fn=pad_custom_sequence_test)
                    acc = 0.0
                    out = {}
                    reslabel = []
                    with torch.no_grad():
                        model = model.eval()
                        torch.cuda.empty_cache()
                        for count, batch in enumerate(data_loader_eval, 0):

                            node_sets = batch[0].to(device)
                            masks = batch[1]
                            word_weight = batch[2].to(device)
                            filename = batch[3]

                            mask = torch.sum(masks, dim=-1)
                            mask = mask  # - 2  # we remove both <s> and </s> symbol
                            mask_onehot = creat_mask(mask)
                            masks = masks.to(device)
                            mask_onehot = mask_onehot.to(device)
                            mask = mask.to(device)
                            with torch.no_grad():
                                embfeat, cls = model.textbertmodel(node_sets, masks)
                                hyp, _ = model.forward(embfeat, cls, word_weight, mask_onehot, mask)

                            for f in range(len(filename)):
                                name = filename[f].split('_')[0]
                                out.update({name: {}})
                                out[name].update({'logprob': hyp[f]})
                                out[name].update({'predict': torch.round(hyp[f].cpu().detach())})

                        with open(os.path.join(Resultsnewdir, "output.pkl"), "wb") as f:
                            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
                        break

                    break

    elif stream == 'image':
        # Start print
        print('--------------------------------')
        # K-fold Cross Validation model evaluation
        # Print
        for fold, (train_ids, dev_ids) in enumerate(kfold.split(dataset.train_dataset)):
            evalacc_best = 0
            evalloss_best = np.Inf
            early_wait = 4
            run_wait = 1
            continuescore = 0
            stop_counter = 0
            ## make model
            model = E2Eimage(768, 4, conf, device)
            # model = torch.nn.DataParallel(model)
            model = model.to(device)

            # optimizer = get_std_opt(model, conf['adim'], int(0.9 * train_examples_len / TRAIN_BATCH_SIZE) * 5, 0.02)
            optimizer = AdamW(model.parameters(),
                              lr=lr,
                              eps=1e-8, weight_decay=0.01
                              )
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=int(
                                                            0.9 * train_examples_len / TRAIN_BATCH_SIZE) * 4,
                                                        num_training_steps=int(
                                                            0.9 * train_examples_len / TRAIN_BATCH_SIZE) * 50)

            Resultsnewdir = os.path.join(Resultsdir, str(fold))
            if not os.path.exists(Resultsnewdir):
                os.makedirs(Resultsnewdir)
            modelnewdir = os.path.join(Savedir, str(fold))
            if not os.path.exists(modelnewdir):
                os.makedirs(modelnewdir)
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            dev_subsampler = torch.utils.data.SubsetRandomSampler(dev_ids)
            dev_examples_len = len(dev_subsampler)
            print(dev_examples_len)

            data_loader_train = torch.utils.data.DataLoader(dataset.train_dataset, batch_size=TRAIN_BATCH_SIZE,
                                                            num_workers=NWORKER, drop_last=True,
                                                            sampler=train_subsampler,
                                                            collate_fn=pad_custom_image_sequence)

            data_loader_dev = torch.utils.data.DataLoader(dataset.train_dataset, batch_size=DEV_BATCH_SIZE,
                                                          num_workers=NWORKER, drop_last=True,
                                                          sampler=dev_subsampler,
                                                          collate_fn=pad_custom_image_sequence)

            criterion = Multilabelsloss(0.1)
            for epoch in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
                if iffinetune == 'false':
                    for param in model.vitnet.parameters():
                        # for param in model.module.BERT.parameters():
                        param.requires_grad = False
                else:
                    for param in model.vitnet.parameters():
                        # for param in model.module.BERT.parameters():
                        param.requires_grad = True

                torch.cuda.empty_cache()
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                model.train()
                trainpredict = []
                trainlabel = []
                for count, batch in enumerate(data_loader_train, 0):
                    image = batch[0].to(device)
                    label = batch[1].to(device)

                    logits, _ = model.forward(image)

                    loss = criterion(logits, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    print("\r%f" % loss, end='')

                    tr_loss += loss.item()
                    nb_tr_steps += 1
                    trainpredict.extend(torch.round(logits.cpu().detach()).tolist())
                    trainlabel.extend(label.cpu().data.numpy().tolist())

                results = []
                total_occurences = 0
                for index in range(1, 5):
                    label = []
                    predict = []
                    for i in range(len(trainlabel)):
                        label.extend([trainlabel[i][index]])
                        if index == 0:
                            predict.extend([int(sum(trainpredict[i]) > 0)])
                        else:
                            predict.extend([trainpredict[i][index - 1]])
                    f1_score = compute_f1(predict, label)
                    f1weight = label.count(True)
                    total_occurences += f1weight
                    results.append(f1_score * f1weight)
                trainallscore = sum(results) / total_occurences

                ## Early Stopping
                print('Early Stoppong check')
                torch.cuda.empty_cache()
                evallossvec = []
                evalacc = 0
                model.eval()
                evalpredict = []
                evalresults = []
                evallabel = []
                for dev_step, dev_batch in enumerate(data_loader_dev):
                    image = dev_batch[0].to(device)
                    label = dev_batch[1].to(device)

                    with torch.no_grad():
                        dev_logits, _ = model.forward(image)

                    evalresults.append(dev_logits.cpu().data.numpy())
                    dev_loss = criterion(dev_logits, label)

                    evallossvec.append(dev_loss.cpu().data.numpy())
                    evalpredict.extend(torch.round(dev_logits.cpu().detach()).tolist())
                    evallabel.extend(label.cpu().data.numpy().tolist())
                results = []
                total_occurences = 0
                for index in range(1, 5):
                    label = []
                    predict = []
                    for i in range(len(evallabel)):
                        label.extend([evallabel[i][index]])
                        if index == 0:
                            predict.extend([int(sum(evalpredict[i]) > 0)])
                        else:
                            predict.extend([evalpredict[i][index - 1]])
                    f1_score = compute_f1(predict, label)
                    f1weight = label.count(True)
                    total_occurences += f1weight
                    results.append(f1_score * f1weight)
                allscore = sum(results) / total_occurences
                evalresults = (np.array(evalresults)).reshape(-1, 1)
                evallabel = (np.array(evallabel)).reshape(-1, 1)
                np.savetxt(os.path.join(Resultsnewdir, 'results' + str(epoch) + '.txt'), evalresults, fmt='%.3f',
                           delimiter=' ',
                           newline='\n', header='', footer='', comments='# ',
                           encoding=None)
                np.savetxt(os.path.join(Resultsnewdir, 'label' + str(epoch) + '.txt'), evallabel, fmt='%.3f',
                           delimiter=' ',
                           newline='\n', header='', footer='', comments='# ',
                           encoding=None)
                evallossmean = np.mean(np.array(evallossvec))
                for param_group in optimizer.param_groups:
                    currentlr = param_group['lr']
                OUTPUT_DIR = os.path.join(modelnewdir,
                                          str(epoch) + '_' + str(evallossmean) + '_' + str(currentlr) + '_' + str(
                                              trainallscore)[:6] + '_' + str(
                                              allscore)[:6] + '.pkl')
                if not os.path.exists(Savedir):
                    os.makedirs(Savedir)
                torch.save(model, OUTPUT_DIR)

                torch.cuda.empty_cache()
                if allscore < evalacc_best:
                    stop_counter = stop_counter + 1
                    print('no improvement')
                    continuescore = 0
                else:
                    print('new score')
                    evalacc_best = allscore
                    continuescore = continuescore + 1

                if continuescore >= run_wait:
                    stop_counter = 0
                print(stop_counter)
                print(early_wait)
                if stop_counter < early_wait:
                    pass
                else:
                    break

            netlist = os.listdir(os.path.join(modelnewdir))
            netlist.sort()
            net_dict = {}
            if best_model == 'loss':
                for m in range(len(netlist)):
                    templist = netlist[m].split('_')
                    net_dict[templist[1]] = netlist[m]
                net_dict = collections.OrderedDict(sorted(net_dict.items(), reverse=True))
            else:
                for m in range(len(netlist)):
                    templist = netlist[m].split('_')
                    net_dict[templist[4]] = netlist[m]
                net_dict = collections.OrderedDict(sorted(net_dict.items()))

            if N_AVERAGE >= 2:
                avg = None
                for n in range(N_AVERAGE):
                    netname = net_dict.get(list(net_dict)[-(n + 1)])
                    print(netname)
                    net = torch.load(os.path.join(modelnewdir, netname)).state_dict()
                    if avg is None:
                        avg = net
                    else:
                        for k in avg.keys():
                            avg[k] += net[k]
                for k in avg.keys():
                    if avg[k] is not None:
                        if avg[k].is_floating_point():
                            avg[k] /= 2
                        else:
                            avg[k] //= 2
                model.load_state_dict(avg)
                torch.save(model, (os.path.join(modelnewdir, 'modelavg2.pkl')))
            else:
                netname = net_dict.get(list(net_dict)[-1])

            Resultsnewdir = os.path.join(Resultsdir, str(fold))
            model = torch.load(os.path.join(Savedir, str(fold), 'modelavg2.pkl'))
            data_loader_eval = torch.utils.data.DataLoader(dataset.test_dataset, batch_size=EVAL_BATCH_SIZE,
                                                           shuffle=False,
                                                           drop_last=False, num_workers=NWORKER,
                                                           collate_fn=pad_custom_image_sequence_test)
            acc = 0.0
            out = {}
            reslabel = []
            with torch.no_grad():
                model = model.eval()
                torch.cuda.empty_cache()
                for count, batch in enumerate(data_loader_eval, 0):
                    image = batch[0].to(device)
                    filename = batch[1]

                    with torch.no_grad():
                        hyp, _ = model.forward(image)

                    for f in range(len(filename)):
                        name = filename[f].split('_')[0]
                        out.update({name: {}})
                        out[name].update({'logprob': hyp[f]})
                        out[name].update({'predict': torch.round(hyp[f].cpu().detach())})

                with open(os.path.join(Resultsnewdir, "output.pkl"), "wb") as f:
                    pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    elif '_bi' in stream:
        print('--------------------------------')
        # K-fold Cross Validation model evaluation
        for fold, (train_ids, dev_ids) in enumerate(kfold.split(dataset.train_dataset)):
            if fold < int(start):
                pass
            else:
                textmdoel = stream.split('_')[0]
                modelResultsdir = os.path.join(Resultsdir)
                if not os.path.exists(modelResultsdir):
                    os.makedirs(modelResultsdir)
                modelSavedir = os.path.join(Savedir)
                if not os.path.exists(modelSavedir):
                    os.makedirs(modelSavedir)
                textmodeldir = os.path.join(mainSavedir, tasktype, 'ocr', textmdoel, str(fold), 'modelavg2.pkl')
                imagemodeldir = os.path.join(mainSavedir, tasktype, 'image', str(fold), 'modelavg2.pkl')
                # Print
                print(f'FOLD {fold}')
                print('--------------------------------')
                evalacc_best = 0
                evalloss_best = np.Inf
                early_wait = 4
                run_wait = 1
                continuescore = 0
                stop_counter = 0

                model = E2Estreamweightsep(1024, 4, conf, device, textmdoel)
                self_state = model.state_dict()
                loaded_text1_state = torch.load(textmodeldir, map_location=device).state_dict()
                loaded_text1_state = {f'textmodel.{k}': v for k, v in loaded_text1_state.items() if
                                      f'textmodel.{k}' in self_state}
                self_state.update(loaded_text1_state)

                loaded_image_state = torch.load(imagemodeldir, map_location=device).state_dict()
                loaded_image_state = {f'Image.{k}': v for k, v in loaded_image_state.items() if
                                      f'Image.{k}' in self_state}
                self_state.update(loaded_image_state)
                model.load_state_dict(self_state)
                model = model.to(device)

                train_examples_len = len(train_ids)
                optimizer = AdamW(model.parameters(),
                                  lr=finetunelr,
                                  eps=1e-8, weight_decay=0.01
                                  )
                scheduler = get_linear_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=int(
                                                                0.9 * train_examples_len / FINE_TUNING_TRAIN_BATCH_SIZE) * 4,
                                                            num_training_steps=int(
                                                                0.9 * train_examples_len / FINE_TUNING_TRAIN_BATCH_SIZE) * 50)

                Resultsnewdir = os.path.join(modelResultsdir, str(fold))
                if not os.path.exists(Resultsnewdir):
                    os.makedirs(Resultsnewdir)
                modelnewdir = os.path.join(modelSavedir, str(fold))
                if not os.path.exists(modelnewdir):
                    os.makedirs(modelnewdir)
                train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
                dev_subsampler = torch.utils.data.SubsetRandomSampler(dev_ids)
                dev_examples_len = len(dev_subsampler)
                print(dev_examples_len)

                data_loader_train = torch.utils.data.DataLoader(dataset.train_dataset,
                                                                batch_size=FINE_TUNING_TRAIN_BATCH_SIZE,
                                                                num_workers=NWORKER, drop_last=True,
                                                                sampler=train_subsampler,
                                                                collate_fn=pad_custom_bi_sequence)

                data_loader_dev = torch.utils.data.DataLoader(dataset.train_dataset,
                                                              batch_size=FINE_TUNING_DEV_BATCH_SIZE,
                                                              num_workers=NWORKER, drop_last=True,
                                                              sampler=dev_subsampler,
                                                              collate_fn=pad_custom_bi_sequence)

                criterion = Multilabelsloss(0.1)
                for epoch in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
                    for param in model.textmodel.parameters():
                        param.requires_grad = False
                    for param in model.Image.parameters():
                        param.requires_grad = False

                    torch.cuda.empty_cache()
                    tr_loss = 0
                    nb_tr_examples, nb_tr_steps = 0, 0
                    model.train()
                    trainpredict = []
                    trainlabel = []
                    for count, batch in enumerate(data_loader_train, 0):
                        textnode_sets = batch[0].to(device)
                        textmasks = batch[1]
                        textword_weight = batch[2].to(device)
                        image = batch[3].to(device)
                        label = batch[4].to(device)

                        textmask = torch.sum(textmasks, dim=-1)
                        textmask_onehot = creat_mask(textmask)
                        textmasks = textmasks.to(device)
                        textmask_onehot = textmask_onehot.to(device)
                        textmask = textmask.to(device)

                        embfeat, cls = model.textmodel.textbertmodel(textnode_sets, textmasks)

                        logits = model.forward(embfeat, cls, textword_weight, \
                                               textmask_onehot, \
                                               textmask, image)

                        loss = criterion(logits, label)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        print("\r%f" % loss, end='')

                        tr_loss += loss.item()
                        nb_tr_steps += 1
                        trainpredict.extend(torch.round(logits.cpu().detach()).tolist())
                        trainlabel.extend(label.cpu().data.numpy().tolist())

                    results = []
                    total_occurences = 0
                    for index in range(1, 5):
                        label = []
                        predict = []
                        for i in range(len(trainlabel)):
                            label.extend([trainlabel[i][index]])
                            if index == 0:
                                predict.extend([int(sum(trainpredict[i]) > 0)])
                            else:
                                predict.extend([trainpredict[i][index - 1]])
                        f1_score = compute_f1(predict, label)
                        f1weight = label.count(True)
                        total_occurences += f1weight
                        results.append(f1_score * f1weight)
                    trainallscore = sum(results) / total_occurences
                    
                    ## Early Stopping
                    print('Early Stoppong check')
                    torch.cuda.empty_cache()
                    evallossvec = []
                    evalacc = 0
                    model.eval()
                    evalpredict = []
                    evalresults = []
                    evallabel = []
                    for dev_step, dev_batch in enumerate(data_loader_dev):
                        textnode_sets = dev_batch[0].to(device)
                        textmasks = dev_batch[1]
                        textword_weight = dev_batch[2].to(device)
                        image = dev_batch[3].to(device)
                        label = dev_batch[4].to(device)

                        textmask = torch.sum(textmasks, dim=-1)
                        textmask_onehot = creat_mask(textmask)
                        textmasks = textmasks.to(device)
                        textmask_onehot = textmask_onehot.to(device)
                        textmask = textmask.to(device)

                        with torch.no_grad():
                            embfeat, cls = model.textmodel.textbertmodel(textnode_sets, textmasks)
                            dev_logits = model.forward(embfeat, cls, textword_weight, \
                                                       textmask_onehot, \
                                                       textmask, image)

                        evalresults.append(dev_logits.cpu().data.numpy())
                        dev_loss = criterion(dev_logits, label)

                        evallossvec.append(dev_loss.cpu().data.numpy())
                        evalpredict.extend(torch.round(dev_logits.cpu().detach()).tolist())
                        evallabel.extend(label.cpu().data.numpy().tolist())
                    results = []
                    total_occurences = 0
                    for index in range(1, 5):
                        label = []
                        predict = []
                        for i in range(len(evallabel)):
                            label.extend([evallabel[i][index]])
                            if index == 0:
                                predict.extend([int(sum(evalpredict[i]) > 0)])
                            else:
                                predict.extend([evalpredict[i][index - 1]])
                        f1_score = compute_f1(predict, label)
                        f1weight = label.count(True)
                        total_occurences += f1weight
                        results.append(f1_score * f1weight)
                    allscore = sum(results) / total_occurences

                    evalresults = (np.array(evalresults)).reshape(-1, 1)
                    evallabel = (np.array(evallabel)).reshape(-1, 1)
                    np.savetxt(os.path.join(Resultsnewdir, 'results' + str(epoch) + '.txt'), evalresults,
                               fmt='%.3f',
                               delimiter=' ',
                               newline='\n', header='', footer='', comments='# ',
                               encoding=None)
                    np.savetxt(os.path.join(Resultsnewdir, 'label' + str(epoch) + '.txt'), evallabel, fmt='%.3f',
                               delimiter=' ',
                               newline='\n', header='', footer='', comments='# ',
                               encoding=None)
                    evallossmean = np.mean(np.array(evallossvec))
                    for param_group in optimizer.param_groups:
                        currentlr = param_group['lr']
                    OUTPUT_DIR = os.path.join(modelnewdir,
                                              str(epoch) + '_' + str(evallossmean) + '_' + str(
                                                  currentlr) + '_' + str(trainallscore)[:6] + '_' + str(
                                                  allscore)[:6] + '.pkl')
                    if not os.path.exists(Savedir):
                        os.makedirs(Savedir)
                    torch.save(model, OUTPUT_DIR)

                    torch.cuda.empty_cache()
                    if allscore < evalacc_best:
                        stop_counter = stop_counter + 1
                        print('no improvement')
                        continuescore = 0
                    else:
                        print('new score')
                        evalacc_best = allscore
                        continuescore = continuescore + 1

                    if continuescore >= run_wait:
                        stop_counter = 0
                    print(stop_counter)
                    print(early_wait)
                    if stop_counter < early_wait:
                        pass
                    else:
                        break

                netlist = os.listdir(os.path.join(modelnewdir))
                netlist.sort()
                net_dict = {}
                if best_model == 'loss':
                    for m in range(len(netlist)):
                        templist = netlist[m].split('_')
                        net_dict[templist[1]] = netlist[m]
                    net_dict = collections.OrderedDict(sorted(net_dict.items(), reverse=True))
                else:
                    for m in range(len(netlist)):
                        templist = netlist[m].split('_')
                        net_dict[templist[4]] = netlist[m]
                    net_dict = collections.OrderedDict(sorted(net_dict.items()))

                if N_AVERAGE >= 2:
                    avg = None
                    for n in range(N_AVERAGE):
                        netname = net_dict.get(list(net_dict)[-(n + 1)])
                        print(netname)
                        net = torch.load(os.path.join(modelnewdir, netname)).state_dict()
                        if avg is None:
                            avg = net
                        else:
                            for k in avg.keys():
                                avg[k] += net[k]
                    for k in avg.keys():
                        if avg[k] is not None:
                            if avg[k].is_floating_point():
                                avg[k] /= 2
                            else:
                                avg[k] //= 2
                    model.load_state_dict(avg)
                    torch.save(model, (os.path.join(modelnewdir, 'modelavg2.pkl')))
                else:
                    netname = net_dict.get(list(net_dict)[-1])

                model = torch.load(os.path.join(modelnewdir, 'modelavg2.pkl'))
                data_loader_eval = torch.utils.data.DataLoader(dataset.test_dataset,
                                                               batch_size=FINE_TUNING_EVAL_BATCH_SIZE,
                                                               shuffle=False,
                                                               drop_last=False, num_workers=NWORKER,
                                                               collate_fn=pad_custom_bi_sequence_test)
                acc = 0.0
                out = {}
                reslabel = []
                with torch.no_grad():
                    model = model.eval()
                    torch.cuda.empty_cache()
                    for count, batch in enumerate(data_loader_eval, 0):
                        textnode_sets = batch[0].to(device)
                        textmasks = batch[1]
                        textword_weight = batch[2].to(device)
                        image = batch[3].to(device)
                        filename = batch[4]

                        textmask = torch.sum(textmasks, dim=-1)
                        textmask_onehot = creat_mask(textmask)
                        textmasks = textmasks.to(device)
                        textmask_onehot = textmask_onehot.to(device)
                        textmask = textmask.to(device)

                        with torch.no_grad():
                            embfeat, cls = model.textmodel.textbertmodel(textnode_sets, textmasks)
                            hyp = model.forward(embfeat, cls, textword_weight, \
                                                textmask_onehot, \
                                                textmask, image)

                        for f in range(len(filename)):
                            name = filename[f].split('_')[0]
                            out.update({name: {}})
                            out[name].update({'logprob': hyp[f]})
                            out[name].update({'predict': torch.round(hyp[f].cpu().detach())})

                    with open(os.path.join(Resultsnewdir, "output.pkl"), "wb") as f:
                        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
                    break
                break

    elif stream == 'bi_BERTC_GCAN':

        print('--------------------------------')
        # K-fold Cross Validation model evaluation
        for fold, (train_ids, dev_ids) in enumerate(kfold.split(dataset.train_dataset)):
            if fold < int(start):
                pass
            else:
                modelResultsdir = os.path.join(Resultsdir)
                if not os.path.exists(modelResultsdir):
                    os.makedirs(modelResultsdir)
                modelSavedir = os.path.join(Savedir)
                if not os.path.exists(modelSavedir):
                    os.makedirs(modelSavedir)
                textgcanmodeldir = os.path.join(mainSavedir, tasktype, 'ocr', 'GCAN', str(fold), 'modelavg2.pkl')
                textbertcmodeldir = os.path.join(mainSavedir, tasktype, 'ocr', 'BERTC', str(fold), 'modelavg2.pkl')
                # Print
                print(f'FOLD {fold}')
                print('--------------------------------')
                evalacc_best = 0
                evalloss_best = np.Inf
                early_wait = 4
                run_wait = 1
                continuescore = 0
                stop_counter = 0

                model = E2EBERTC_GCAN(1024, 4, conf, device)

                self_state = model.state_dict()
                loaded_text1_state = torch.load(textgcanmodeldir, map_location=device).state_dict()
                loaded_text1_state = {f'GCAN.{k}': v for k, v in loaded_text1_state.items() if
                                      f'GCAN.{k}' in self_state}
                self_state.update(loaded_text1_state)

                loaded_text2_state = torch.load(textbertcmodeldir, map_location=device).state_dict()
                loaded_text2_state = {f'BERTC.{k}': v for k, v in loaded_text2_state.items() if
                                      f'BERTC.{k}' in self_state}
                self_state.update(loaded_text2_state)
                model.load_state_dict(self_state)
                model = model.to(device)

                train_examples_len = len(train_ids)
                optimizer = AdamW(model.parameters(),
                                  lr=finetunelr,
                                  eps=1e-8, weight_decay=0.01
                                  )
                scheduler = get_linear_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=int(
                                                                0.9 * train_examples_len / FINE_TUNING_TRAIN_BATCH_SIZE) * 4,
                                                            num_training_steps=int(
                                                                0.9 * train_examples_len / FINE_TUNING_TRAIN_BATCH_SIZE) * 50)

                Resultsnewdir = os.path.join(modelResultsdir, str(fold))
                if not os.path.exists(Resultsnewdir):
                    os.makedirs(Resultsnewdir)
                modelnewdir = os.path.join(modelSavedir, str(fold))
                if not os.path.exists(modelnewdir):
                    os.makedirs(modelnewdir)
                train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
                dev_subsampler = torch.utils.data.SubsetRandomSampler(dev_ids)
                dev_examples_len = len(dev_subsampler)
                print(dev_examples_len)

                data_loader_train = torch.utils.data.DataLoader(dataset.train_dataset,
                                                                batch_size=FINE_TUNING_TRAIN_BATCH_SIZE,
                                                                num_workers=NWORKER, drop_last=True,
                                                                sampler=train_subsampler,
                                                                collate_fn=pad_custom_bi_sequence)

                data_loader_dev = torch.utils.data.DataLoader(dataset.train_dataset,
                                                              batch_size=FINE_TUNING_DEV_BATCH_SIZE,
                                                              num_workers=NWORKER, drop_last=True,
                                                              sampler=dev_subsampler,
                                                              collate_fn=pad_custom_bi_sequence)

                criterion = Multilabelsloss(0.1)
                for epoch in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
                    for param in model.BERTC.parameters():
                        param.requires_grad = False
                    for param in model.GCAN.parameters():
                        param.requires_grad = False

                    torch.cuda.empty_cache()
                    tr_loss = 0
                    nb_tr_examples, nb_tr_steps = 0, 0
                    model.train()
                    trainpredict = []
                    trainlabel = []
                    for count, batch in enumerate(data_loader_train, 0):
                        textnode_sets = batch[0].to(device)
                        textmasks = batch[1]
                        textword_weight = batch[2].to(device)
                        image = batch[3].to(device)
                        label = batch[4].to(device)

                        textmask = torch.sum(textmasks, dim=-1)
                        textmask_onehot = creat_mask(textmask)
                        textmasks = textmasks.to(device)
                        textmask_onehot = textmask_onehot.to(device)
                        textmask = textmask.to(device)


                        embfeatb, clsb = model.BERTC.textbertmodel(textnode_sets, textmasks)
                        embfeatg, clsg = model.GCAN.textbertmodel(textnode_sets, textmasks)

                        logits = model.forward(embfeatb, embfeatg, clsb, clsg, textword_weight, \
                                               textmask_onehot, \
                                               textmask, image)

                        loss = criterion(logits, label)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        print("\r%f" % loss, end='')

                        tr_loss += loss.item()
                        nb_tr_steps += 1
                        trainpredict.extend(torch.round(logits.cpu().detach()).tolist())
                        trainlabel.extend(label.cpu().data.numpy().tolist())

                    results = []
                    total_occurences = 0
                    for index in range(1, 5):
                        label = []
                        predict = []
                        for i in range(len(trainlabel)):
                            label.extend([trainlabel[i][index]])
                            if index == 0:
                                predict.extend([int(sum(trainpredict[i]) > 0)])
                            else:
                                predict.extend([trainpredict[i][index - 1]])
                        f1_score = compute_f1(predict, label)
                        f1weight = label.count(True)
                        total_occurences += f1weight
                        results.append(f1_score * f1weight)
                    trainallscore = sum(results) / total_occurences
                    ## Early Stopping
                    print('Early Stoppong check')
                    torch.cuda.empty_cache()
                    evallossvec = []
                    evalacc = 0
                    model.eval()
                    evalpredict = []
                    evalresults = []
                    evallabel = []
                    for dev_step, dev_batch in enumerate(data_loader_dev):
                        textnode_sets = dev_batch[0].to(device)
                        textmasks = dev_batch[1]
                        textword_weight = dev_batch[2].to(device)
                        image = dev_batch[3].to(device)
                        label = dev_batch[4].to(device)

                        textmask = torch.sum(textmasks, dim=-1)
                        textmask_onehot = creat_mask(textmask)
                        textmasks = textmasks.to(device)
                        textmask_onehot = textmask_onehot.to(device)
                        textmask = textmask.to(device)

                        with torch.no_grad():
                            embfeatb, clsb = model.BERTC.textbertmodel(textnode_sets, textmasks)
                            embfeatg, clsg = model.GCAN.textbertmodel(textnode_sets, textmasks)
                            dev_logits = model.forward(embfeatb, embfeatg, clsb, clsg, textword_weight, \
                                                       textmask_onehot, \
                                                       textmask, image)

                        evalresults.append(dev_logits.cpu().data.numpy())
                        dev_loss = criterion(dev_logits, label)

                        evallossvec.append(dev_loss.cpu().data.numpy())
                        evalpredict.extend(torch.round(dev_logits.cpu().detach()).tolist())
                        evallabel.extend(label.cpu().data.numpy().tolist())
                    results = []
                    total_occurences = 0
                    for index in range(1, 5):
                        label = []
                        predict = []
                        for i in range(len(evallabel)):
                            label.extend([evallabel[i][index]])
                            if index == 0:
                                predict.extend([int(sum(evalpredict[i]) > 0)])
                            else:
                                predict.extend([evalpredict[i][index - 1]])
                        f1_score = compute_f1(predict, label)
                        f1weight = label.count(True)
                        total_occurences += f1weight
                        results.append(f1_score * f1weight)
                    allscore = sum(results) / total_occurences

                    evalresults = (np.array(evalresults)).reshape(-1, 1)
                    evallabel = (np.array(evallabel)).reshape(-1, 1)
                    np.savetxt(os.path.join(Resultsnewdir, 'results' + str(epoch) + '.txt'), evalresults,
                               fmt='%.3f',
                               delimiter=' ',
                               newline='\n', header='', footer='', comments='# ',
                               encoding=None)
                    np.savetxt(os.path.join(Resultsnewdir, 'label' + str(epoch) + '.txt'), evallabel, fmt='%.3f',
                               delimiter=' ',
                               newline='\n', header='', footer='', comments='# ',
                               encoding=None)
                    evallossmean = np.mean(np.array(evallossvec))
                    for param_group in optimizer.param_groups:
                        currentlr = param_group['lr']
                    OUTPUT_DIR = os.path.join(modelnewdir,
                                              str(epoch) + '_' + str(evallossmean) + '_' + str(
                                                  currentlr) + '_' + str(trainallscore)[:6] + '_' + str(
                                                  allscore)[:6] + '.pkl')
                    if not os.path.exists(Savedir):
                        os.makedirs(Savedir)
                    torch.save(model, OUTPUT_DIR)

                    torch.cuda.empty_cache()
                    if allscore < evalacc_best:
                        stop_counter = stop_counter + 1
                        print('no improvement')
                        continuescore = 0
                    else:
                        print('new score')
                        evalacc_best = allscore
                        continuescore = continuescore + 1

                    if continuescore >= run_wait:
                        stop_counter = 0
                    print(stop_counter)
                    print(early_wait)
                    if stop_counter < early_wait:
                        pass
                    else:
                        break

                netlist = os.listdir(os.path.join(modelnewdir))
                netlist.sort()
                net_dict = {}
                if best_model == 'loss':
                    for m in range(len(netlist)):
                        templist = netlist[m].split('_')
                        net_dict[templist[1]] = netlist[m]
                    net_dict = collections.OrderedDict(sorted(net_dict.items(), reverse=True))
                else:
                    for m in range(len(netlist)):
                        templist = netlist[m].split('_')
                        net_dict[templist[4]] = netlist[m]
                    net_dict = collections.OrderedDict(sorted(net_dict.items()))

                if N_AVERAGE >= 2:
                    avg = None
                    for n in range(N_AVERAGE):
                        netname = net_dict.get(list(net_dict)[-(n + 1)])
                        print(netname)
                        net = torch.load(os.path.join(modelnewdir, netname), map_location='cpu').state_dict()
                        if avg is None:
                            avg = net
                        else:
                            for k in avg.keys():
                                avg[k] += net[k]
                    for k in avg.keys():
                        if avg[k] is not None:
                            if avg[k].is_floating_point():
                                avg[k] /= 2
                            else:
                                avg[k] //= 2
                    model.load_state_dict(avg)
                    torch.save(model, (os.path.join(modelnewdir, 'modelavg2.pkl')))
                else:
                    netname = net_dict.get(list(net_dict)[-1])
                del model
                model = torch.load(os.path.join(modelnewdir, 'modelavg2.pkl'))
                data_loader_eval = torch.utils.data.DataLoader(dataset.test_dataset,
                                                               batch_size=FINE_TUNING_EVAL_BATCH_SIZE,
                                                               shuffle=False,
                                                               drop_last=False, num_workers=NWORKER,
                                                               collate_fn=pad_custom_bi_sequence_test)
                acc = 0.0
                out = {}
                reslabel = []
                with torch.no_grad():
                    model = model.eval()
                    torch.cuda.empty_cache()
                    for count, batch in enumerate(data_loader_eval, 0):
                        textnode_sets = batch[0].to(device)
                        textmasks = batch[1]
                        textword_weight = batch[2].to(device)
                        image = batch[3].to(device)
                        filename = batch[4]

                        textmask = torch.sum(textmasks, dim=-1)
                        textmask_onehot = creat_mask(textmask)
                        textmasks = textmasks.to(device)
                        textmask_onehot = textmask_onehot.to(device)
                        textmask = textmask.to(device)

                        with torch.no_grad():
                            embfeatb, clsb = model.BERTC.textbertmodel(textnode_sets, textmasks)
                            embfeatg, clsg = model.GCAN.textbertmodel(textnode_sets, textmasks)
                            hyp = model.forward(embfeatb, embfeatg, clsb, clsg, textword_weight, \
                                                textmask_onehot, \
                                                textmask, image)

                        for f in range(len(filename)):
                            name = filename[f].split('_')[0]
                            out.update({name: {}})
                            out[name].update({'logprob': hyp[f]})
                            out[name].update({'predict': torch.round(hyp[f].cpu().detach())})

                    with open(os.path.join(Resultsnewdir, "output.pkl"), "wb") as f:
                        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
                    break
                break

    elif 'bi' in stream:
        print('--------------------------------')
        # K-fold Cross Validation model evaluation
        for fold, (train_ids, dev_ids) in enumerate(kfold.split(dataset.train_dataset)):
            if fold < int(start):
                pass
            else:
                modelResultsdir = os.path.join(Resultsdir)
                if not os.path.exists(modelResultsdir):
                    os.makedirs(modelResultsdir)
                modelSavedir = os.path.join(Savedir)
                if not os.path.exists(modelSavedir):
                    os.makedirs(modelSavedir)
                textgcanmodeldir = os.path.join(mainSavedir, tasktype, 'ocr', 'GCAN', str(fold), 'modelavg2.pkl')
                textbertcmodeldir = os.path.join(mainSavedir, tasktype, 'ocr', 'BERTC', str(fold), 'modelavg2.pkl')
                imagemodeldir = os.path.join(mainSavedir, tasktype, 'image', str(fold), 'modelavg2.pkl')
                # Print
                print(f'FOLD {fold}')
                print('--------------------------------')
                evalacc_best = 0
                evalloss_best = np.Inf
                early_wait = 4
                run_wait = 1
                continuescore = 0
                stop_counter = 0

                model = E2Eall(1024, 4, conf, device)

                self_state = model.state_dict()
                loaded_text1_state = torch.load(textgcanmodeldir, map_location=device).state_dict()
                loaded_text1_state = {f'GCAN.{k}': v for k, v in loaded_text1_state.items() if
                                      f'GCAN.{k}' in self_state}
                self_state.update(loaded_text1_state)

                loaded_text2_state = torch.load(textbertcmodeldir, map_location=device).state_dict()
                loaded_text2_state = {f'BERTC.{k}': v for k, v in loaded_text2_state.items() if
                                      f'BERTC.{k}' in self_state}
                self_state.update(loaded_text2_state)

                loaded_image_state = torch.load(imagemodeldir, map_location=device).state_dict()
                loaded_image_state = {f'Image.{k}': v for k, v in loaded_image_state.items() if
                                      f'Image.{k}' in self_state}
                self_state.update(loaded_image_state)

                model.load_state_dict(self_state)
                model = model.to(device)

                train_examples_len = len(train_ids)
                # optimizer = get_std_opt(model, conf['adim'], int(0.9 * train_examples_len / TRAIN_BATCH_SIZE) * 5, 0.02)
                optimizer = AdamW(model.parameters(),
                                  lr=finetunelr,
                                  eps=1e-8, weight_decay=0.01
                                  )
                scheduler = get_linear_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=int(
                                                                0.9 * train_examples_len / FINE_TUNING_TRAIN_BATCH_SIZE) * 4,
                                                            num_training_steps=int(
                                                                0.9 * train_examples_len / FINE_TUNING_TRAIN_BATCH_SIZE) * 50)

                Resultsnewdir = os.path.join(modelResultsdir, str(fold))
                if not os.path.exists(Resultsnewdir):
                    os.makedirs(Resultsnewdir)
                modelnewdir = os.path.join(modelSavedir, str(fold))
                if not os.path.exists(modelnewdir):
                    os.makedirs(modelnewdir)
                train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
                dev_subsampler = torch.utils.data.SubsetRandomSampler(dev_ids)
                dev_examples_len = len(dev_subsampler)
                print(dev_examples_len)

                data_loader_train = torch.utils.data.DataLoader(dataset.train_dataset,
                                                                batch_size=FINE_TUNING_TRAIN_BATCH_SIZE,
                                                                num_workers=NWORKER, drop_last=True,
                                                                sampler=train_subsampler,
                                                                collate_fn=pad_custom_bi_sequence)

                data_loader_dev = torch.utils.data.DataLoader(dataset.train_dataset,
                                                              batch_size=FINE_TUNING_DEV_BATCH_SIZE,
                                                              num_workers=NWORKER, drop_last=True,
                                                              sampler=dev_subsampler,
                                                              collate_fn=pad_custom_bi_sequence)

                criterion = Multilabelsloss(0.1)
                for epoch in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
                    for param in model.BERTC.parameters():
                        param.requires_grad = False
                    for param in model.GCAN.parameters():
                        param.requires_grad = False
                    for param in model.Image.parameters():
                        param.requires_grad = False


                    torch.cuda.empty_cache()
                    tr_loss = 0
                    nb_tr_examples, nb_tr_steps = 0, 0
                    model.train()
                    trainpredict = []
                    trainlabel = []
                    for count, batch in enumerate(data_loader_train, 0):
                        textnode_sets = batch[0].to(device)
                        textmasks = batch[1]
                        textword_weight = batch[2].to(device)
                        image = batch[3].to(device)
                        label = batch[4].to(device)

                        textmask = torch.sum(textmasks, dim=-1)
                        textmask_onehot = creat_mask(textmask)
                        textmasks = textmasks.to(device)
                        textmask_onehot = textmask_onehot.to(device)
                        textmask = textmask.to(device)


                        embfeatb, clsb = model.BERTC.textbertmodel(textnode_sets, textmasks)
                        embfeatg, clsg = model.GCAN.textbertmodel(textnode_sets, textmasks)

                        logits = model.forward(embfeatb, embfeatg, clsb, clsg, textword_weight, \
                                               textmask_onehot, \
                                               textmask, image)

                        loss = criterion(logits, label)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        print("\r%f" % loss, end='')

                        tr_loss += loss.item()
                        nb_tr_steps += 1
                        trainpredict.extend(torch.round(logits.cpu().detach()).tolist())
                        trainlabel.extend(label.cpu().data.numpy().tolist())

                    results = []
                    total_occurences = 0
                    for index in range(1, 5):
                        label = []
                        predict = []
                        for i in range(len(trainlabel)):
                            label.extend([trainlabel[i][index]])
                            if index == 0:
                                predict.extend([int(sum(trainpredict[i]) > 0)])
                            else:
                                predict.extend([trainpredict[i][index - 1]])
                        f1_score = compute_f1(predict, label)
                        f1weight = label.count(True)
                        total_occurences += f1weight
                        results.append(f1_score * f1weight)
                    trainallscore = sum(results) / total_occurences

                    ## Early Stopping
                    print('Early Stoppong check')
                    torch.cuda.empty_cache()
                    evallossvec = []
                    evalacc = 0
                    model.eval()
                    evalpredict = []
                    evalresults = []
                    evallabel = []
                    for dev_step, dev_batch in enumerate(data_loader_dev):
                        textnode_sets = dev_batch[0].to(device)
                        textmasks = dev_batch[1]
                        textword_weight = dev_batch[2].to(device)
                        image = dev_batch[3].to(device)
                        label = dev_batch[4].to(device)

                        textmask = torch.sum(textmasks, dim=-1)
                        textmask_onehot = creat_mask(textmask)
                        textmasks = textmasks.to(device)
                        textmask_onehot = textmask_onehot.to(device)
                        textmask = textmask.to(device)

                        with torch.no_grad():
                            embfeatb, clsb = model.BERTC.textbertmodel(textnode_sets, textmasks)
                            embfeatg, clsg = model.GCAN.textbertmodel(textnode_sets, textmasks)
                            dev_logits = model.forward(embfeatb, embfeatg, clsb, clsg, textword_weight, \
                                                       textmask_onehot, \
                                                       textmask, image)

                        evalresults.append(dev_logits.cpu().data.numpy())
                        dev_loss = criterion(dev_logits, label)

                        evallossvec.append(dev_loss.cpu().data.numpy())
                        evalpredict.extend(torch.round(dev_logits.cpu().detach()).tolist())
                        evallabel.extend(label.cpu().data.numpy().tolist())
                    results = []
                    total_occurences = 0
                    for index in range(1, 5):
                        label = []
                        predict = []
                        for i in range(len(evallabel)):
                            label.extend([evallabel[i][index]])
                            if index == 0:
                                predict.extend([int(sum(evalpredict[i]) > 0)])
                            else:
                                predict.extend([evalpredict[i][index - 1]])
                        f1_score = compute_f1(predict, label)
                        f1weight = label.count(True)
                        total_occurences += f1weight
                        results.append(f1_score * f1weight)
                    allscore = sum(results) / total_occurences

                    evalresults = (np.array(evalresults)).reshape(-1, 1)
                    evallabel = (np.array(evallabel)).reshape(-1, 1)
                    np.savetxt(os.path.join(Resultsnewdir, 'results' + str(epoch) + '.txt'), evalresults,
                               fmt='%.3f',
                               delimiter=' ',
                               newline='\n', header='', footer='', comments='# ',
                               encoding=None)
                    np.savetxt(os.path.join(Resultsnewdir, 'label' + str(epoch) + '.txt'), evallabel, fmt='%.3f',
                               delimiter=' ',
                               newline='\n', header='', footer='', comments='# ',
                               encoding=None)
                    evallossmean = np.mean(np.array(evallossvec))
                    for param_group in optimizer.param_groups:
                        currentlr = param_group['lr']
                    OUTPUT_DIR = os.path.join(modelnewdir,
                                              str(epoch) + '_' + str(evallossmean) + '_' + str(
                                                  currentlr) + '_' + str(trainallscore)[:6] + '_' + str(
                                                  allscore)[:6] + '.pkl')
                    if not os.path.exists(Savedir):
                        os.makedirs(Savedir)
                    torch.save(model, OUTPUT_DIR)

                    torch.cuda.empty_cache()
                    if allscore < evalacc_best:
                        stop_counter = stop_counter + 1
                        print('no improvement')
                        continuescore = 0
                    else:
                        print('new score')
                        evalacc_best = allscore
                        continuescore = continuescore + 1

                    if continuescore >= run_wait:
                        stop_counter = 0
                    print(stop_counter)
                    print(early_wait)
                    if stop_counter < early_wait:
                        pass
                    else:
                        break

                netlist = os.listdir(os.path.join(modelnewdir))
                netlist.sort()
                net_dict = {}
                if best_model == 'loss':
                    for m in range(len(netlist)):
                        templist = netlist[m].split('_')
                        net_dict[templist[1]] = netlist[m]
                    net_dict = collections.OrderedDict(sorted(net_dict.items(), reverse=True))
                else:
                    for m in range(len(netlist)):
                        templist = netlist[m].split('_')
                        net_dict[templist[4]] = netlist[m]
                    net_dict = collections.OrderedDict(sorted(net_dict.items()))

                if N_AVERAGE >= 2:
                    avg = None
                    for n in range(N_AVERAGE):
                        netname = net_dict.get(list(net_dict)[-(n + 1)])
                        print(netname)
                        net = torch.load(os.path.join(modelnewdir, netname), map_location='cpu').state_dict()
                        if avg is None:
                            avg = net
                        else:
                            for k in avg.keys():
                                avg[k] += net[k]
                    for k in avg.keys():
                        if avg[k] is not None:
                            if avg[k].is_floating_point():
                                avg[k] /= 2
                            else:
                                avg[k] //= 2
                    model.load_state_dict(avg)
                    torch.save(model, (os.path.join(modelnewdir, 'modelavg2.pkl')))
                else:
                    netname = net_dict.get(list(net_dict)[-1])
                del model
                model = torch.load(os.path.join(modelnewdir, 'modelavg2.pkl'))
                data_loader_eval = torch.utils.data.DataLoader(dataset.test_dataset,
                                                               batch_size=FINE_TUNING_EVAL_BATCH_SIZE,
                                                               shuffle=False,
                                                               drop_last=False, num_workers=NWORKER,
                                                               collate_fn=pad_custom_bi_sequence_test)
                acc = 0.0
                out = {}
                reslabel = []
                with torch.no_grad():
                    model = model.eval()
                    torch.cuda.empty_cache()
                    for count, batch in enumerate(data_loader_eval, 0):
                        textnode_sets = batch[0].to(device)
                        textmasks = batch[1]
                        textword_weight = batch[2].to(device)
                        image = batch[3].to(device)
                        filename = batch[4]

                        textmask = torch.sum(textmasks, dim=-1)
                        textmask_onehot = creat_mask(textmask)
                        textmasks = textmasks.to(device)
                        textmask_onehot = textmask_onehot.to(device)
                        textmask = textmask.to(device)

                        with torch.no_grad():
                            embfeatb, clsb = model.BERTC.textbertmodel(textnode_sets, textmasks)
                            embfeatg, clsg = model.GCAN.textbertmodel(textnode_sets, textmasks)
                            hyp = model.forward(embfeatb, embfeatg, clsb, clsg, textword_weight, \
                                                textmask_onehot, \
                                                textmask, image)


                        for f in range(len(filename)):
                            name = filename[f].split('_')[0]
                            out.update({name: {}})
                            out[name].update({'logprob': hyp[f]})
                            out[name].update({'predict': torch.round(hyp[f].cpu().detach())})

                    with open(os.path.join(Resultsnewdir, "output.pkl"), "wb") as f:
                        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
                    break
                break


if __name__ == '__main__':
    train(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10], sys.argv[11])





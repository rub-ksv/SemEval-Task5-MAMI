import torch
import kaldiio
import os
from torch_scatter import scatter_add
from collections import defaultdict
import scipy.sparse as sp
from tqdm import tqdm
from math import log
from PIL import Image
import numpy as np
from sklearn import metrics
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch.utils.data import DataLoader, Dataset

def check_matrix(matrix, gold, pred):
  """Check matrix dimension."""
  if matrix.size == 1:
    tmp = matrix[0][0]
    matrix = np.zeros((2, 2))
    if (pred[1] == 0):
      if gold[1] == 0:  #true negative
        matrix[0][0] = tmp
      else:  #falsi negativi
        matrix[1][0] = tmp
    else:
      if gold[1] == 0:  #false positive
        matrix[0][1] = tmp
      else:  #true positive
        matrix[1][1] = tmp
  return matrix

def compute_f1(pred_values, gold_values):
  matrix = metrics.confusion_matrix(gold_values, pred_values)
  matrix = check_matrix(matrix, gold_values, pred_values)

  #positive label
  if matrix[0][0] == 0:
    pos_precision = 0.0
    pos_recall = 0.0
  else:
    pos_precision = matrix[0][0] / (matrix[0][0] + matrix[0][1])
    pos_recall = matrix[0][0] / (matrix[0][0] + matrix[1][0])

  if (pos_precision + pos_recall) != 0:
    pos_F1 = 2 * (pos_precision * pos_recall) / (pos_precision + pos_recall)
  else:
    pos_F1 = 0.0

  #negative label
  neg_matrix = [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]

  if neg_matrix[0][0] == 0:
    neg_precision = 0.0
    neg_recall = 0.0
  else:
    neg_precision = neg_matrix[0][0] / (neg_matrix[0][0] + neg_matrix[0][1])
    neg_recall = neg_matrix[0][0] / (neg_matrix[0][0] + neg_matrix[1][0])

  if (neg_precision + neg_recall) != 0:
    neg_F1 = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall)
  else:
    neg_F1 = 0.0

  f1 = (pos_F1 + neg_F1) / 2
  return f1

def pad_custom_sequence(sequences):
    node_sets_sequence = []
    masks_sequence = []
    word_weight_sequence = []
    label_sequence = []
    for node_sets, masks, word_weight, label in sequences:
        node_sets_sequence.append(node_sets.squeeze(0))
        masks_sequence.append(masks.squeeze(0))
        word_weight_sequence.append(word_weight)
        label_sequence.append(label)
    node_sets_sequence = torch.nn.utils.rnn.pad_sequence(node_sets_sequence, batch_first=True, padding_value=1)
    masks_sequence = torch.nn.utils.rnn.pad_sequence(masks_sequence, batch_first=True, padding_value=0)
    word_weight_sequence, _ = padding_tensor(word_weight_sequence, padding_idx=0.0)
    label_sequence = torch.nn.utils.rnn.pad_sequence(label_sequence, batch_first=True)
    return node_sets_sequence, masks_sequence, word_weight_sequence, label_sequence

def pad_custom_sequence_test(sequences):
    node_sets_sequence = []
    masks_sequence = []
    word_weight_sequence = []
    filename_sequence = []
    for node_sets, masks, word_weight, filename in sequences:
        node_sets_sequence.append(node_sets.squeeze(0))
        masks_sequence.append(masks.squeeze(0))
        word_weight_sequence.append(word_weight)
        filename_sequence.append(filename)
    node_sets_sequence = torch.nn.utils.rnn.pad_sequence(node_sets_sequence, batch_first=True, padding_value=1)
    masks_sequence = torch.nn.utils.rnn.pad_sequence(masks_sequence, batch_first=True, padding_value=0)
    word_weight_sequence, _ = padding_tensor(word_weight_sequence, padding_idx=0.0)
    return node_sets_sequence, masks_sequence, word_weight_sequence, filename_sequence


def pad_custom_image_sequence(sequences):
    image_sequence = []
    label_sequence = []
    for image, label in sequences:
        image_sequence.append(image)
        label_sequence.append(label)
    image_sequence = torch.nn.utils.rnn.pad_sequence(image_sequence, batch_first=True, padding_value=0)
    label_sequence = torch.nn.utils.rnn.pad_sequence(label_sequence, batch_first=True)
    return image_sequence, label_sequence

def pad_custom_image_sequence_test(sequences):
    image_sequence = []
    filename_sequence = []
    for image, label in sequences:
        image_sequence.append(image)
        filename_sequence.append(label)
    image_sequence = torch.nn.utils.rnn.pad_sequence(image_sequence, batch_first=True, padding_value=0)
    return image_sequence, filename_sequence


def pad_custom_bi_sequence(sequences):
    text_node_sets_sequence = []
    text_masks_sequence = []
    text_word_weight_sequence = []
    image_sequence = []
    label_sequence = []
    for text_node_sets, text_masks, text_word_weight, image, label in sequences:
        text_node_sets_sequence.append(text_node_sets.squeeze(0))
        text_masks_sequence.append(text_masks.squeeze(0))
        text_word_weight_sequence.append(text_word_weight)
        image_sequence.append(image)
        label_sequence.append(label)
    text_node_sets_sequence = torch.nn.utils.rnn.pad_sequence(text_node_sets_sequence, batch_first=True, padding_value=1)
    text_masks_sequence = torch.nn.utils.rnn.pad_sequence(text_masks_sequence, batch_first=True, padding_value=0)
    text_word_weight_sequence, _ = padding_tensor(text_word_weight_sequence, padding_idx=0.0)
    image_sequence = torch.nn.utils.rnn.pad_sequence(image_sequence, batch_first=True, padding_value=0)
    label_sequence = torch.nn.utils.rnn.pad_sequence(label_sequence, batch_first=True)
    return text_node_sets_sequence, text_masks_sequence, \
           text_word_weight_sequence, image_sequence, label_sequence

def pad_custom_bi_sequence_test(sequences):
    text_node_sets_sequence = []
    text_masks_sequence = []
    text_word_weight_sequence = []
    image_sequence = []
    filename_sequence = []
    for text_node_sets, text_masks, text_word_weight, image, label in sequences:
        text_node_sets_sequence.append(text_node_sets.squeeze(0))
        text_masks_sequence.append(text_masks.squeeze(0))
        text_word_weight_sequence.append(text_word_weight)
        image_sequence.append(image)
        filename_sequence.append(label)
    text_node_sets_sequence = torch.nn.utils.rnn.pad_sequence(text_node_sets_sequence, batch_first=True, padding_value=1)
    text_masks_sequence = torch.nn.utils.rnn.pad_sequence(text_masks_sequence, batch_first=True, padding_value=0)
    text_word_weight_sequence, _ = padding_tensor(text_word_weight_sequence, padding_idx=0.0)
    image_sequence = torch.nn.utils.rnn.pad_sequence(image_sequence, batch_first=True, padding_value=0)
    return text_node_sets_sequence, text_masks_sequence, \
           text_word_weight_sequence, image_sequence, filename_sequence

def creat_mask(mask):
    maxLen = torch.max(mask).to(mask.device)
    mask_onehot = torch.arange(maxLen).expand(len(mask), maxLen) < mask.unsqueeze(1)
    mask_onehot = mask_onehot.type(torch.FloatTensor)
    return mask_onehot

def padding_tensor(sequences, padding_idx=1):
    num = len(sequences)
    max_len_0 = max([s.shape[0] for s in sequences])
    max_len_1 = max([s.shape[1] for s in sequences])
    out_dims = (num, max_len_0, max_len_1)
    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_idx)
    for i, tensor in enumerate(sequences):
        len_0 = tensor.size(0)
        len_1 = tensor.size(1)
        out_tensor[i, :len_0, :len_1] = tensor
    mask = out_tensor == padding_idx # Marking all places with padding_idx as mask
    return out_tensor, mask

class TextLevelGNNDataset(Dataset):  # For instantiating train, validation and test dataset
    def __init__(self, datadict, tasktype, stream, dset, text_out_word_id_map=None, text_out_doc_id_map=None,
                 text_adj_max=None, transform=None):
        super(TextLevelGNNDataset).__init__()
        self.datakeys = self._get_keys(datadict)
        self.datadict = datadict
        self.text_out_word_id_map = text_out_word_id_map
        self.text_out_doc_id_map = text_out_doc_id_map
        self.text_adj_max = text_adj_max
        self.tasktype = tasktype
        self.dset = dset

        self.stream = stream
        if transform is None:
            pass
        else:
            self.transform = transform

    def _get_keys(self, datadict):
        """Return absolute paths to all utterances, transcriptions and phoneme labels in the required subset."""
        keys = list(datadict.keys())
        return keys

    def __len__(self):
        return len(self.datakeys)

    def __getitem__(self, index):
        if 'ocr' in self.stream:
            twid = self.datadict[self.datakeys[index]]['textids']
            twmask = self.datadict[self.datakeys[index]]['text_attention_mask']
            filename = self.datakeys[index]

            doc_id = self.text_out_doc_id_map[filename]
            word_id = [self.text_out_word_id_map[id] for id in twid.squeeze(0).tolist()]
            ids = [doc_id]
            ids.extend(word_id[1:])
            word_weight = torch.zeros([len(ids), len(ids)])
            for i in range(len(ids)):
                for j in range(len(ids)):
                    word_weight[i][j] = self.text_adj_max[(ids[i], ids[j])]
            word_weight = torch.FloatTensor(word_weight)

            if self.dset == 'training':
                if self.tasktype == 'misogynous':
                    label = int(self.datadict[self.datakeys[index]][self.tasktype])
                    label = torch.FloatTensor([label])
                else:
                    label1 = int(self.datadict[self.datakeys[index]]['misogynous'])
                    label2 = int(self.datadict[self.datakeys[index]]['shaming'])
                    label3 = int(self.datadict[self.datakeys[index]]['stereotype'])
                    label4 = int(self.datadict[self.datakeys[index]]['objectification'])
                    label5 = int(self.datadict[self.datakeys[index]]['violence'])
                    label = torch.FloatTensor([label1, label2, label3, label4, label5])
                return torch.LongTensor(twid), torch.LongTensor(twmask), \
                       word_weight, label
            else:
                return torch.LongTensor(twid), torch.LongTensor(twmask), \
                       word_weight, filename
        elif 'image' in self.stream:
            imagedir = self.datadict[self.datakeys[index]]['image']#.replace('./', './../')
            filename = self.datakeys[index]
            img = Image.open(imagedir)

            if img.mode != 'RGB':
                img = img.convert('RGB')
            transpic = self.transform(img)  # .unsqueeze(0)
            std, mean = torch.std_mean(transpic.view(3, -1), dim=-1)
            transpic = (transpic - mean[:, None, None]) / std[:, None, None]

            if self.dset == 'training':
                if self.tasktype == 'misogynous':
                    label = int(self.datadict[self.datakeys[index]][self.tasktype])
                    label = torch.FloatTensor([label])
                else:
                    label1 = int(self.datadict[self.datakeys[index]]['misogynous'])
                    label2 = int(self.datadict[self.datakeys[index]]['shaming'])
                    label3 = int(self.datadict[self.datakeys[index]]['stereotype'])
                    label4 = int(self.datadict[self.datakeys[index]]['objectification'])
                    label5 = int(self.datadict[self.datakeys[index]]['violence'])
                    label = torch.FloatTensor([label1, label2, label3, label4, label5])
                return transpic, label
            else:
                return transpic, filename

        elif 'bi' in self.stream:
            texttwid = self.datadict[self.datakeys[index]]['textids']
            texttwmask = self.datadict[self.datakeys[index]]['text_attention_mask']
            filename = self.datakeys[index]

            text_doc_id = self.text_out_doc_id_map[filename]
            text_word_id = [self.text_out_word_id_map[id] for id in texttwid.squeeze(0).tolist()]
            text_ids = [text_doc_id]
            text_ids.extend(text_word_id[1:])
            text_word_weight = torch.zeros([len(text_ids), len(text_ids)])
            for i in range(len(text_ids)):
                for j in range(len(text_ids)):
                    text_word_weight[i][j] = self.text_adj_max[(text_ids[i], text_ids[j])]
            text_word_weight = torch.FloatTensor(text_word_weight)

            imagedir = self.datadict[self.datakeys[index]]['image']#.replace('./', './../')
            img = Image.open(imagedir)

            if img.mode != 'RGB':
                img = img.convert('RGB')
            transpic = self.transform(img)  # .unsqueeze(0)
            std, mean = torch.std_mean(transpic.view(3, -1), dim=-1)
            transpic = (transpic - mean[:, None, None]) / std[:, None, None]

            if self.dset == 'training':
                if self.tasktype == 'misogynous':
                    label = int(self.datadict[self.datakeys[index]][self.tasktype])
                    label = torch.FloatTensor([label])
                else:
                    label1 = int(self.datadict[self.datakeys[index]]['misogynous'])
                    label2 = int(self.datadict[self.datakeys[index]]['shaming'])
                    label3 = int(self.datadict[self.datakeys[index]]['stereotype'])
                    label4 = int(self.datadict[self.datakeys[index]]['objectification'])
                    label5 = int(self.datadict[self.datakeys[index]]['violence'])
                    label = torch.FloatTensor([label1, label2, label3, label4, label5])

                return torch.LongTensor(texttwid), \
                       torch.LongTensor(texttwmask), \
                       text_word_weight, transpic, label

            else:
                return torch.LongTensor(texttwid), \
                       torch.LongTensor(texttwmask), \
                       text_word_weight, transpic, filename




class TextLevelGNNDatasetClass:  # This class is used to achieve parameters sharing among datasets
    def __init__(self, train_file, test_file, tokenizer, tasktype, MAX_LENGTH=10, p=2, min_freq=2,
                 train_validation_split=None, stream=None, transform=None, adjdir=None):
        self.tokenizer = tokenizer
        self.tasktype = tasktype
        self.MAX_LENGTH = MAX_LENGTH
        self.p = p
        self.min_freq = min_freq
        self.train_validation_split = train_validation_split
        self.transform = transform
        self.stream = stream
        self.adjdir = adjdir


        self.stoi = {}  # Re-index
        self.itos = {}  # Re-index
        self.vocab_count = len(self.stoi)
        self.embedding_matrix = None

        if train_validation_split is not None:
            self.train_dataset, self.validation_dataset = random_split(self.train_data.to_numpy(),
                                                                   [int(len(self.train_data) * train_validation_split),
                                                                    len(self.train_data) - int(
                                                                        len(self.train_data) * train_validation_split)])
        self.test_file = test_file
        self.train_file = train_file

        self.build_vocab()  # Based on train_dataset only. Updates self.stoi, self.itos, self.vocab_count and self.embedding_matrix

        self.train_dataset, self.test_dataset = self.prepare_dataset()

    def build_vocab(self):
        if 'ocr' in self.stream:
            [self.train_file[pic].update(
                {'textids': self.tokenizer(self.train_file[pic]['text'], return_tensors='pt').data['input_ids']})
                for pic in self.train_file]
            [self.train_file[pic].update({'text_attention_mask':
                                              self.tokenizer(self.train_file[pic]['text'],
                                                             return_tensors='pt').data[
                                                  'attention_mask']})
             for pic in self.train_file]
            [self.test_file[pic].update(
                {'textids': self.tokenizer(self.test_file[pic]['text'], return_tensors='pt').data['input_ids']})
                for pic in self.test_file]
            [self.test_file[pic].update({'text_attention_mask':
                                             self.tokenizer(self.test_file[pic]['text'], return_tensors='pt').data[
                                                 'attention_mask']})
             for pic in self.test_file]
            if not os.path.exists(os.path.join(self.adjdir, 'adj_matrix.pt')):
                self.vocab_text_dict = defaultdict(str)
                for pic in self.train_file:
                    #picname = int(pic.split('.')[0])
                    self.vocab_text_dict[pic] = self.tokenizer(self.train_file[pic]['text']).data['input_ids']
                for pic in self.test_file:
                    #picname = int(pic.split('.')[0])
                    self.vocab_text_dict[pic] = self.tokenizer(self.test_file[pic]['text']).data['input_ids']

                word_freq = get_vocab(self.vocab_text_dict)
                vocab = list(word_freq.keys())
                words_in_docs, word_doc_freq = build_word_doc_edges(self.vocab_text_dict)
                word_id_map = {word: i for i, word in enumerate(vocab)}
                sparse_graph, self.text_out_doc_id_map, self.text_out_word_id_map = build_edges(self.vocab_text_dict, word_id_map,
                                                                                      vocab,
                                                                                      word_doc_freq, self.p)
                A = sparse_graph.tocoo()
                row = torch.from_numpy(A.row).to(torch.long)
                col = torch.from_numpy(A.col).to(torch.long)
                edge_index = torch.stack([row, col], dim=0)
                edge_weight = torch.from_numpy(A.data).to(torch.float)
                edge_weight = edge_weight.view(-1)
                assert edge_weight.size(0) == edge_index.size(1)

                num_nodes = len(self.text_out_doc_id_map.keys()) + len(self.text_out_word_id_map.keys())

                edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
                loop_weight = torch.full((num_nodes,),
                                         2,
                                         dtype=edge_weight.dtype)
                edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

                row = edge_index[0]
                col = edge_index[1]
                deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

                A_tel = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

                self.text_doc_A_tel = defaultdict(float)
                for i, weight in enumerate(A_tel):
                    row_id = row[i]
                    col_id = col[i]
                    doc_id_pair = (int(row_id), int(col_id))
                    self.text_doc_A_tel[doc_id_pair] = weight

                torch.save(self.text_doc_A_tel, os.path.join(self.adjdir, 'adj_matrix.pt'))
                torch.save(self.text_out_doc_id_map, os.path.join(self.adjdir, 'doc_id.pt'))
                torch.save(self.text_out_word_id_map, os.path.join(self.adjdir, 'word_id.pt'))
            else:
                self.text_doc_A_tel = torch.load(os.path.join(self.adjdir, 'adj_matrix.pt'))
                self.text_out_doc_id_map = torch.load(os.path.join(self.adjdir, 'doc_id.pt'))
                self.text_out_word_id_map = torch.load(os.path.join(self.adjdir, 'word_id.pt'))

        elif 'image' in self.stream:
            pass

        elif 'bi' in self.stream:
            [self.train_file[pic].update(
                {'textids': self.tokenizer(self.train_file[pic]['text'], return_tensors='pt').data['input_ids']})
                for pic in self.train_file]
            [self.train_file[pic].update({'text_attention_mask':
                                              self.tokenizer(self.train_file[pic]['text'],
                                                             return_tensors='pt').data[
                                                  'attention_mask']})
             for pic in self.train_file]
            [self.test_file[pic].update(
                {'textids': self.tokenizer(self.test_file[pic]['text'], return_tensors='pt').data['input_ids']})
                for pic in self.test_file]
            [self.test_file[pic].update({'text_attention_mask':
                                             self.tokenizer(self.test_file[pic]['text'], return_tensors='pt').data[
                                                 'attention_mask']})
             for pic in self.test_file]

            [self.train_file[pic].update(
                {'imageids': self.tokenizer(self.train_file[pic]['imagetext'], return_tensors='pt').data[
                    'input_ids']})
                for pic in self.train_file]
            [self.train_file[pic].update({'image_attention_mask':
                                              self.tokenizer(self.train_file[pic]['imagetext'],
                                                             return_tensors='pt').data[
                                                  'attention_mask']})
             for pic in self.train_file]
            [self.test_file[pic].update(
                {'imageids': self.tokenizer(self.test_file[pic]['imagetext'], return_tensors='pt').data[
                    'input_ids']})
                for pic in self.test_file]
            [self.test_file[pic].update({'image_attention_mask':
                                             self.tokenizer(self.test_file[pic]['imagetext'],
                                                            return_tensors='pt').data[
                                                 'attention_mask']})
             for pic in self.test_file]
            self.text_doc_A_tel = torch.load(os.path.join(self.adjdir, 'adj_matrix.pt'))
            self.text_out_doc_id_map = torch.load(os.path.join(self.adjdir, 'doc_id.pt'))
            self.text_out_word_id_map = torch.load(os.path.join(self.adjdir, 'word_id.pt'))





    def prepare_dataset(self):  # will also build self.edge_stat and self.public_edge_mask
        if 'ocr' in self.stream:
            train_dataset = TextLevelGNNDataset(self.train_file, \
                                                self.tasktype, self.stream, 'training', \
                                                text_out_word_id_map=self.text_out_word_id_map, \
                                                text_out_doc_id_map=self.text_out_doc_id_map, \
                                                text_adj_max=self.text_doc_A_tel, \
                                                transform=self.transform)
            # preparing self.test_dataset
            test_dataset = TextLevelGNNDataset(self.test_file, \
                                               self.tasktype, self.stream, 'test', \
                                               text_out_word_id_map=self.text_out_word_id_map, \
                                               text_out_doc_id_map=self.text_out_doc_id_map, \
                                               text_adj_max=self.text_doc_A_tel, \
                                               transform=self.transform)
        elif 'image' in self.stream:
            train_dataset = TextLevelGNNDataset(self.train_file, \
                                                self.tasktype, self.stream, 'training',\
                                                transform=self.transform)
            # preparing self.test_dataset
            test_dataset = TextLevelGNNDataset(self.test_file, \
                                               self.tasktype, self.stream, 'test',\
                                               transform=self.transform)
        else:
            train_dataset = TextLevelGNNDataset(self.train_file, \
                                                self.tasktype, self.stream, 'training',\
                                                text_out_word_id_map=self.text_out_word_id_map, \
                                                text_out_doc_id_map=self.text_out_doc_id_map, \
                                                text_adj_max=self.text_doc_A_tel, \
                                                transform=self.transform)
            # preparing self.test_dataset
            test_dataset = TextLevelGNNDataset(self.test_file, \
                                               self.tasktype, self.stream,  'test',\
                                               text_out_word_id_map=self.text_out_word_id_map, \
                                               text_out_doc_id_map=self.text_out_doc_id_map, \
                                               text_adj_max=self.text_doc_A_tel, \
                                               transform=self.transform)


        return train_dataset, test_dataset


def get_vocab(text_dict):
    word_freq = defaultdict(int)
    keys = text_dict.keys()
    for key in keys:
        text = text_dict[key]
        for word in text:
            word_freq[word] += 1
    return word_freq

def build_word_doc_edges(doc_dict):
    # build all docs that a word is contained in
    words_in_docs = defaultdict(set)
    for i, doc_id in enumerate(doc_dict):
        words = doc_dict[doc_id]
        for word in words:
            words_in_docs[word].add(i)

    word_doc_freq = {}
    for word, doc_list in words_in_docs.items():
        word_doc_freq[word] = len(doc_list)

    return words_in_docs, word_doc_freq

def build_edges(doc_dict, word_id_map, vocab, word_doc_freq, window_size=20):
    # constructing all windows
    windows = []
    for doc_id in doc_dict.keys():
        words = doc_dict[doc_id]
        doc_length = len(words)
        if doc_length <= window_size:
            windows.append(words)
        else:
            for i in range(doc_length - window_size + 1):
                window = words[i: i + window_size]
                windows.append(window)
    # constructing all single word frequency
    word_window_freq = defaultdict(int)
    for window in windows:
        appeared = set()
        for word in window:
            if word not in appeared:
                word_window_freq[word] += 1
                appeared.add(word)
    # constructing word pair count frequency
    word_pair_count = defaultdict(int)
    for window in tqdm(windows):
        for i in range(1, len(window)):
            for j in range(i):
                word_i = window[i]
                word_j = window[j]
                word_i_id = word_id_map[word_i]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_count[(word_i_id, word_j_id)] += 1
                word_pair_count[(word_j_id, word_i_id)] += 1
    row = []
    col = []
    weight = []

    # pmi as weights
    num_docs = len(doc_dict.keys())
    num_window = len(windows)
    for word_id_pair, count in tqdm(word_pair_count.items()):
        i, j = word_id_pair[0], word_id_pair[1]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(num_docs + i)
        col.append(num_docs + j)
        weight.append(pmi)

    out_doc_id_map = {doc_id: i for i, doc_id in enumerate(doc_dict)}
    out_word_id_map = {doc_id: word_id_map[doc_id] + num_docs for i, doc_id in enumerate(word_id_map)}
    # frequency of document word pair
    doc_word_freq = defaultdict(int)
    for i, doc_id in enumerate(doc_dict):
        words = doc_dict[doc_id]
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = (i, word_id)
            doc_word_freq[doc_word_str] += 1

    for i, doc_id in enumerate(doc_dict):
        words = doc_dict[doc_id]
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            word_id = word_id_map[word]
            freq = doc_word_freq[(i, word_id)]
            row.append(i)
            col.append(num_docs + word_id)
            idf = log(1.0 * num_docs /
                      word_doc_freq[vocab[word_id]])
            weight.append(freq * idf)
            doc_word_set.add(word)

    number_nodes = num_docs + len(vocab)
    adj_mat = sp.csr_matrix((weight, (row, col)), shape=(number_nodes, number_nodes))
    adj = adj_mat + adj_mat.T.multiply(adj_mat.T > adj_mat) - adj_mat.multiply(adj_mat.T > adj_mat)
    return adj, out_doc_id_map, out_word_id_map

import torch
from trans.layer_norm import LayerNorm
import torch.nn.functional as F
import math
import timm
from transformers import RobertaTokenizer, RobertaModel


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class MultiSequential(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""

    def forward(self, *args):
        """Repeat."""
        for m in self:
            args = m(*args)
        return args
def repeat(N, fn):
    """Repeat module N times.
    Args:
        N (int): Number of repeat time.
        fn (Callable): Function to generate module.
    Returns:
        MultiSequential: Repeated model instance.
    """
    return MultiSequential(*[fn(n) for n in range(N)])

class MessagePassing(torch.nn.Module):
    def __init__(self, input_size, device, n_head, dropout_rate=0, iflast=True):
        super(MessagePassing, self).__init__()
        self.input_size = input_size # d
        self.dropout_rate = dropout_rate
        self.d_k = input_size // n_head
        self.h = n_head
        self.attn = None
        self.linear_q = torch.nn.Linear(input_size, input_size)
        self.linear_k = torch.nn.Linear(input_size, input_size)
        self.linear_v = torch.nn.Linear(input_size, input_size)
        self.linear_out = torch.nn.Linear(input_size, input_size)
        #self.linear = torch.nn.Linear(self.input_size, self.out_size) # (d, c)
        #self.weight = torch.nn.Linear(2 * p + 1, 2 * p + 1)  # (d, c)
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.elu = torch.nn.ELU()
        self.device = device
        self.iflast = iflast
        self.LN = LayerNorm(input_size)
        self.linear_out2 = torch.nn.Linear(input_size, input_size)
        self.lastlinear_out = torch.nn.Linear(self.d_k, input_size)
    def attention(self, embedded_node, key, value, word_weight, iflast=True):
        n_batch = embedded_node.size(0)
        n_seq = embedded_node.size(1)
        q = self.linear_q(embedded_node).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        word_weight_rep = word_weight.unsqueeze(1).repeat(1, self.h, 1, 1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        self.attn = torch.softmax(scores, dim=-1)
        scores = self.dropout(self.attn)
        x = torch.matmul(scores, v)
        x = torch.matmul(word_weight_rep, x)
        out = x.transpose(1, 2)

        if iflast == False:
            x = out.contiguous().view(embedded_node.shape)
            x = F.elu(self.linear_out2(x))
            x = self.LN(x + embedded_node)
        else:
            x = torch.mean(out, dim=-2)
            x = F.elu(self.lastlinear_out(x))
            x = self.LN(x + embedded_node)

        return x

    def forward(self, embedded_node, mask_onehot, word_weight):
        # node_sets: (batch_size, l)
        # embedded_node: (batch_size, l, d)
        # edge_weight: (batch_size, max_sentence_length, max_neighbor_count)
        # embedded_neighbor_node: (batch_size, max_sentence_length, max_neighbor_count, d)
        #neighbor_embedded_node, distance_matrix = self.create_neighbor_set(embedded_node, mask, p=self.p)
        #neighbor_embedded_node = torch.abs(neighbor_embedded_node).to(self.device)
        #distance_matrix = torch.abs(distance_matrix).to(self.device)
        #maskneigh_onehot = torch.ones_like(neighbor_embedded_node).to(self.device)
        #maskneigh_onehot = maskneigh_onehot.masked_fill(
        #    (neighbor_embedded_node == 0),
        #    0)  # (batch_size, max_sentence_length, max_neighbor_count)
        embedded_node = self.attention(embedded_node, embedded_node, embedded_node, word_weight, iflast=self.iflast)

        #mapped = (mask_onehot.view(-1, 1) * embedded_node.view(-1, self.input_size)).view(
        #    mapped.shape)
        return embedded_node, mask_onehot, word_weight

def vitnet():
    pretrained = True
    progress = True
    pretrained_v = timm.create_model('vit_large_patch16_224', pretrained=pretrained)

    return pretrained_v


class AttentionHead(torch.nn.Module):

    def __init__(self, input_dim=768, hidden_dim=512):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, last_hidden_state):
        """
        Note:
        "last_hidden_state" shape is [batch_size, seq_len, 768].
        The "weights" produced from softmax will add up to 1 across all tokens in each record.
        """
        linear1_output = self.linear1(last_hidden_state)  # Shape is [batch_size, seq_len, 512]
        activation = torch.tanh(linear1_output)  # Shape is [batch_size, seq_len, 512]
        score = self.linear2(activation)  # Shape is [batch_size, seq_len, 1]
        weights = torch.softmax(score, dim=1)  # Shape is [batch_size, seq_len, 1]
        result = torch.sum(weights * last_hidden_state, dim=1)  # Shape is [batch_size, 768]
        return result

class E2EBERTC(torch.nn.Module):
    def __init__(self, uttidim, odim, conf, device):
        torch.nn.Module.__init__(self)
        self.device = device
        self.inputdim = conf['adim']
        self.padding_id = conf['padding_index']
        self.BERTtext = RobertaModel.from_pretrained('roberta-large')
        self.textlinear = torch.nn.Sequential(
            torch.nn.Linear(conf['adim'], int(conf['adim'] / 2)),
            torch.nn.ReLU(),
            torch.nn.Dropout(conf['dropout-rate']),
            torch.nn.Linear(int(conf['adim'] / 2), odim)
        )

    def textbertmodel(self, node_sets, masks):
        # 1. forward encoder
        outputs = self.BERTtext(node_sets, masks)
        clsfeat = outputs.pooler_output
        wordxs_pad = outputs['last_hidden_state']
        return wordxs_pad, clsfeat

    def forward(self, embedded_node_input, cls, word_weight, mask_onehot, mask):
        x = self.textlinear(cls)  # (batch_size, c)
        x = F.sigmoid(x)  # (batch_size, c) along the c dimension
        return x, cls


class E2EGCAN(torch.nn.Module):
    def __init__(self, uttidim, odim, conf, device):
        torch.nn.Module.__init__(self)
        self.device = device
        self.inputdim = conf['adim']
        self.padding_id = conf['padding_index']
        self.BERTtext = RobertaModel.from_pretrained('roberta-large')
        self.text_message_passing1 = MessagePassing(input_size=conf['adim'], device=self.device, n_head=conf['nheads'],
                                        dropout_rate=conf['dropout-rate'], iflast=False)
        self.text_message_passing2 = MessagePassing(input_size=conf['adim'], device=self.device, n_head=conf['nheads'],
                                                    dropout_rate=conf['dropout-rate'], iflast=False)
        self.text_message_passing_last = MessagePassing(input_size=conf['adim'], device=self.device, n_head=conf['nheads'],
                                        dropout_rate=conf['dropout-rate'], iflast=True)

        self.textgcanencoder = torch.nn.Sequential(
            torch.nn.Linear(conf['adim'], conf['adim']),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(conf['adim'], conf['adim']),
            torch.nn.ReLU(),
            LayerNorm(conf['adim']),
            torch.nn.Linear(conf['adim'], odim)

        )

    def textbertmodel(self, node_sets, masks):
        # 1. forward encoder
        outputs = self.BERTtext(node_sets, masks)
        clsfeat = outputs.pooler_output
        wordxs_pad = outputs['last_hidden_state']
        return wordxs_pad, clsfeat

    def forward(self, embedded_node_input, cls, word_weight, mask_onehot, mask):
        embedded_node, mask_onehot, word_weight = self.text_message_passing1(embedded_node_input, mask_onehot, word_weight)
        embedded_node, mask_onehot, word_weight = self.text_message_passing2(embedded_node, mask_onehot,
                                                                             word_weight)
        embedded_node_GCAN, mask_onehot, word_weight = self.text_message_passing_last(embedded_node, mask_onehot, word_weight)

        embedded_node_GCAN = embedded_node_GCAN.sum(dim=1)
        x = self.textgcanencoder(embedded_node_GCAN)
        x = F.sigmoid(x)
        return x, embedded_node_GCAN

class E2Eimage(torch.nn.Module):
    def __init__(self, uttidim, odim, conf, device):
        torch.nn.Module.__init__(self)
        self.device = device
        self.inputdim = conf['adim']
        self.vitnet = vitnet()
        #self.message_passing = MessagePassing(vertice_count=vocab_count, input_size=768, out_size=odim, p=3, dropout_rate=conf['dropout-rate'], padding_idx=self.padding_id) # input_size: (d,); out_size: (c,)
        self.vitnet.head = Identity()
        self.imagefeatureencoder = torch.nn.Sequential(
            torch.nn.Linear(conf['adim'], int(conf['adim'] / 2)),
            torch.nn.ReLU(),
            torch.nn.Dropout(conf['dropout-rate']),
            torch.nn.Linear(int(conf['adim'] / 2), odim)
        )


    def forward(self, image):
        imagefeat = self.vitnet(image)  ## image in Vit (Vision transformer), output dim = 1000
        x = self.imagefeatureencoder(imagefeat)
        x = F.sigmoid(x) # (batch_size, c) along the c dimension
        return x, imagefeat

class E2Estreamweightsep(torch.nn.Module):
    def __init__(self, uttidim, odim, conf, device, textmodel):
        torch.nn.Module.__init__(self)
        self.device = device
        if textmodel == 'BERTC':
            self.textmodel = E2EBERTC(uttidim, odim, conf, device)
        elif textmodel == 'GCAN':
            self.textmodel = E2EGCAN(uttidim, odim, conf, device)

        self.Image = E2Eimage(uttidim, odim, conf, device)
        self.textencoder = torch.nn.Sequential(
            torch.nn.Linear(2 * odim + 2 * conf['adim'], conf['adim']),
            torch.nn.ReLU(),
            torch.nn.Linear(conf['adim'], 2)
        )
        self.featencoder = torch.nn.Sequential(
            torch.nn.Linear(2 * odim + 2 * conf['adim'], conf['adim']),
            torch.nn.ReLU(),
            torch.nn.Dropout(conf['dropout-rate']),
            torch.nn.Linear(conf['adim'], odim)

        )

    def forward(self, embedded_model, cls, word_weight, mask_onehot, mask, image):
        prob_modeltext, textfeat = self.textmodel(embedded_model, cls, word_weight, mask_onehot, mask)
        prob_modelimage, imagefeat = self.Image(image)
        probsfeats = torch.cat((prob_modeltext, prob_modelimage, textfeat, imagefeat), dim=-1)

        x = F.softmax(self.textencoder(probsfeats), dim=-1)

        x = prob_modeltext * x[:, 0].unsqueeze(1) + prob_modelimage * x[:, 1].unsqueeze(1)
        x_ffd = F.sigmoid(self.featencoder(probsfeats))
        x = (x + x_ffd) / 2

        return x

class E2EBERTC_GCAN(torch.nn.Module):
    def __init__(self, uttidim, odim, conf, device):
        torch.nn.Module.__init__(self)
        self.device = device
        self.BERTC = E2EBERTC(uttidim, odim, conf, device)
        self.GCAN = E2EGCAN(uttidim, odim, conf, device)

        self.textencoder = torch.nn.Sequential(
            torch.nn.Linear(2 * odim + 2 * conf['adim'], conf['adim']),
            torch.nn.ReLU(),
            torch.nn.Linear(conf['adim'], 2)
        )
        self.featencoder = torch.nn.Sequential(
            torch.nn.Linear(2 * odim + 2 * conf['adim'], conf['adim']),
            torch.nn.ReLU(),
            torch.nn.Dropout(conf['dropout-rate']),
            torch.nn.Linear(conf['adim'], odim)

        )

    def forward(self, embedded_modelb, embedded_modelg, clsb, clsg, word_weight, mask_onehot, mask, image):
        prob_modeltextb, textfeatb = self.BERTC(embedded_modelb, clsb, word_weight, mask_onehot, mask)
        prob_modeltextg, textfeatg = self.GCAN(embedded_modelg, clsg, word_weight, mask_onehot, mask)

        probsfeats = torch.cat(
            (prob_modeltextb, prob_modeltextg, textfeatb, textfeatg), dim=-1)
        x = F.softmax(self.textencoder(probsfeats), dim=-1)
        x = prob_modeltextb * x[:, 0].unsqueeze(1) + prob_modeltextg * x[:, 1].unsqueeze(1)
        x_ffd = F.sigmoid(self.featencoder(probsfeats))
        x = (x + x_ffd) / 2

        return x


class E2Eall(torch.nn.Module):
    ## from paper https://pan.webis.de/downloads/publications/papers/takahashi_2018.pdf
    def __init__(self, uttidim, odim, conf, device):
        torch.nn.Module.__init__(self)
        self.device = device
        self.BERTC = E2EBERTC(uttidim, odim, conf, device)
        self.GCAN = E2EGCAN(uttidim, odim, conf, device)
        self.Image = E2Eimage(uttidim, odim, conf, device)
        #self.cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)


        self.weightencoder = torch.nn.Sequential(
            torch.nn.Conv1d(3, 1, 512, stride=1),
            torch.nn.ReLU(),
            torch.nn.Linear(517, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 3)
        )

        self.probencoder = torch.nn.Sequential(
            torch.nn.Conv1d(3, 1, 512, stride=1),
            torch.nn.ReLU(),
            torch.nn.Linear(517, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, odim)
        )

    def forward(self, embedded_model1, embedded_model2, clsbertc, clsgcan, word_weight, mask_onehot, mask, image):
        prob_modeltext1, text1feat = self.BERTC(embedded_model1, clsbertc, word_weight, mask_onehot, mask)
        prob_modeltext2, text2feat = self.GCAN(embedded_model2, clsgcan, word_weight, mask_onehot, mask)
        prob_modelimage, imagefeat = self.Image(image)
        text1feat = torch.cat((text1feat, prob_modeltext1), dim=-1)
        text2feat = torch.cat((text2feat, prob_modeltext2), dim=-1)
        imgfeat = torch.cat((imagefeat, prob_modelimage), dim=-1)
        featmatrix = torch.cat((text1feat.unsqueeze(1), text2feat.unsqueeze(1), imgfeat.unsqueeze(1)), dim=-2)


        x = F.softmax(self.weightencoder(featmatrix).squeeze(1), dim=-1)
        x = prob_modeltext1 * x[:, 0].unsqueeze(1) + prob_modeltext2 * x[:, 1].unsqueeze(1) + prob_modelimage * x[:, 2].unsqueeze(1)

        fusion = self.probencoder(featmatrix).squeeze(1)

        x_ffd = F.sigmoid(fusion)
        x = (x + x_ffd) / 2
        return x


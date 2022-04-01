import torch
import torch.nn.functional as F
import numpy as np
import json, os, sys
import pickle
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
import imageio
import cv2
import multiprocessing as mp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def caption_image_beam_search(encoder, decoder, image_path, word_map, rev_word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """
    outdict = {}
    imagename = '.'.join(image_path.split('\\')[-1].split('.')[:-1])

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    print(image_path)
    img = imageio.imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)

    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    std, mean = torch.std_mean(img.view(3, -1), dim=-1)
    image = (img - mean[:, None, None]) / std[:, None, None]

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        prev_word_inds = prev_word_inds.long()
        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]
    data = visualize_att(seq, alphas, rev_word_map)
    outdict.update({imagename: data})
    return outdict


def visualize_att(seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """


    words = [rev_word_map[ind] for ind in seq]

    words.remove('<start>')
    words.remove('<end>')
    words = ' '.join(words)
    return words

def product_helper(args):
    return caption_image_beam_search(*args)

def main(imagedir, dset, savedir, maindir, ifgpu):

    if ifgpu == 'true':
        device = 'cuda'
    else:
        device = 'cpu'

    model = os.path.join(maindir, './BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar')
    word_map = os.path.join(maindir, './WORDMAP_coco_5_cap_per_img_5_min_word_freq.json')
    beam_size = 5
    Savedir = os.path.join(savedir, dset)
    if not os.path.exists(Savedir):
        os.makedirs(Savedir)
    # Load model
    autherlist = os.listdir(os.path.join(imagedir, dset + '_split'))
    checkpoint = torch.load(model, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    with open(word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word


    resultdict = {}
    #autherlist.remove(dset + '.csv')
    for author in autherlist:
        resultdict.update({author + '.jpg': {}})
        imgfolder = os.path.join(imagedir, dset + '_split', author)
        imgorgfolder = os.path.join(imagedir, dset, author)
        text = []
        textdict = {}
        imagelist = os.listdir(imgfolder)
        if not imagelist:
            imagedirsub = os.path.join(imgorgfolder + '.jpg')
            word = caption_image_beam_search(encoder, decoder, imagedirsub, word_map, rev_word_map, beam_size)
            text.append(list(word.values())[0])
        else:
            for i in imagelist:
                imagedirsub = os.path.join(imgfolder, i)
                try:
                    word = caption_image_beam_search(encoder, decoder, imagedirsub, word_map, rev_word_map, beam_size)
                    text.append(list(word.values())[0])
                except:
                    pass

        text = list(set(text))
        text = ' and '.join(text)
        textdict.update({os.path.join(imagedir, dset, author): text})
        resultdict[author + '.jpg'].update(textdict)


    with open(os.path.join(Savedir, dset + "_image_to_text.json"), 'w', encoding='utf-8') as json_file:
        json.dump(resultdict, json_file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    # hand over parameter overview
    # sys.argv[1] = imagedir (str): Dataset dir
    # sys.argv[2] = dset (str): Which set.
    # sys.argv[3] = savedir (str): The dir saves caption results
    # sys.argv[4] = maindir (str): Image-caption code dir, where also saves the pre-trained image-caption model.
    # sys.argv[5] = ifgpu (str): If use GPU for caption.
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])



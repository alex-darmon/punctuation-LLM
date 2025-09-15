#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 23:04:01 2018

@author: alexandra.darmon
"""

import pickle
import time
import numpy as np
import pandas as pd
import sys
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
import os
import spacy
from string import ascii_letters
spacy.load('en_core_web_sm')
from spacy.lang.en import English
    
empirical_nb_words = 40
empirical_nb_sentences = 200


punctuation_vector = ['!', '"', '(', ')', ',', '.', ':', ';', '?', '^']
punctuation_end = ['!', '?', '.', '^']
other_pun = set(punctuation_vector) - set(punctuation_end) 
alpha = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
exceptions = ["Mr", "Mrs","Dr","Prof","St", "etc"]
NB_sign = len(punctuation_vector)

freq_pun_col = ['FREQ_'+str(i) for i in range(0,NB_sign)]
freq_nb_words_col = ['FREQ_WORD_'+str(i) for i in range(0,empirical_nb_words)]
freq_length_sen_with_col = ['FREQ_SEN_'+str(i) for i in range(0,empirical_nb_sentences)]
transition_mat_col = ['TRANS_'+str(i) for i in range(0,NB_sign*NB_sign)]          
norm_transition_mat_col = ['NORM_TRANS_'+str(i) for i in range(0,NB_sign*NB_sign)]
mat_nb_words_pun_col = ['MAT_WORD_'+str(i) for i in range(0,NB_sign*NB_sign)]



def ranks_of_freq(freq):
    new_freq = [i for i in freq]
    ranks = np.zeros((1,len(new_freq)), dtype='f')
    for i in range(1,len(new_freq)+1):
        ind = new_freq.index(max(new_freq))
        if max(new_freq) == 0: ranks[0,ind] = None
        else: ranks[0,ind] = i
        new_freq[ind] = -1
    return ranks

def save_as_pickled_object(obj, filepath):
    """
    This is a defensive way to write pickle.write, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])


def try_to_load_as_pickled_object_or_None(filepath):
    """
    This is a defensive way to write pickle.load, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    try:
        input_size = os.path.getsize(filepath)
        bytes_in = bytearray(0)
        with open(filepath, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        obj = pickle.loads(bytes_in)
    except:
        return None
    return obj


def get_gutenberg_text(book_id):
    try:
        text = strip_headers(load_etext(int(book_id))).strip()
        return text
    except:
        return None


def cleaning_punctuation(ch):
    if ch == '...': return '^'
    return ch



def get_tokens_word_nb_punctuation(tokens):
    try:
        tokens_word_nb_punctuation = []
        count = 0
        for token in tokens:
            token = str(token)
            if token in punctuation_vector or token=="'":
                tokens_word_nb_punctuation+= [count, token.replace("'",'"')]
                count = 0
            elif len(set(alpha).intersection(set(str(token))))>0:
                count+=1
        return tokens_word_nb_punctuation
    except:
        return None



def get_textinfo(list_book_ids):
    t_start = time.time()
    df_res = pd.DataFrame(None)
    list_raw_seq_nb_words = []
    error_book_id = []
    count = 0
    for book_id in list_book_ids:
        if count%50==0: 
            print(count)
            print(time.time() - t_start)
        count+=1
        try:
            text = strip_headers(load_etext(int(book_id))).strip()
            parser = English(max_length=len(text)+1)
            tokens = parser(text)
            raw_seq_nb_words = get_tokens_word_nb_punctuation(tokens)
            list_raw_seq_nb_words.append(raw_seq_nb_words)
        except:
            print(book_id)
            error_book_id.append(int(book_id))
            list_raw_seq_nb_words.append(None)
    df_res['book_id'] = list_book_ids
    df_res['seq_nb_words'] = list_raw_seq_nb_words
    print(time.time() - t_start)
    return df_res


def get_frequencies(tokens, vector=punctuation_vector):
    try:
        freqs = []
        for elt in vector:
            freqs.append((tokens.count(elt)))
        return list(map(lambda x: x/ sum(freqs), freqs))
    except:
        return None

def seq_nb_only(seq):
    try:
        if type(seq[0])==int: res= seq[0:-1:2]
        else: res = seq[1::2]
        return res
    except:
        return None

def seq_pun_only(seq): 
    try:
        if type(seq[0])==int: res= seq[1::2]
        else: res = seq[0:-1:2]
        return res
    except:
        return None


def get_tokens_sentences_nb(tokens, include_pun=False):
    try:
        tokens_sentences_nb = []
        count = 0
        for token in tokens:
            if token in punctuation_end:
                tokens_sentences_nb+= [count, token]
                count = 0
            else:
                if token not in punctuation_vector:
                    if type(token) == int:
                        count+=token
        return tokens_sentences_nb
    except:
        return None


def chunks(l, n):
   n = max(1, n)
   return list(l[i:i+n] for i in range(0, len(l), n))


def update_mat_tot(tot,mat,char1,char2):
    ind1 = punctuation_vector.index(char1)
    ind2 = punctuation_vector.index(char2)
    s = tot[ind1]= tot[ind1]+1
    mat[ind1,:] = mat[ind1,:]*(s-1)
    mat[ind1,ind2] = mat[ind1,ind2]+1
    mat[ind1,:] = mat[ind1,:]/s
    
    
#This function gives the transition matrix of a 
#sequence of punctuation marks
def transition_mat(seq_pun):
    try:
        transition_mat =np.zeros((NB_sign, NB_sign), dtype='f')
        count_pun =  np.zeros(NB_sign, dtype='f')
    
        for i in range(0,len(seq_pun)-1):
            update_mat_tot(count_pun,
                           transition_mat,seq_pun[i],seq_pun[i+1])
        return transition_mat
    except:
        return None

# normalized with frequencies
def normalised_transition_mat(mat,freq_pun):
    try:
        res = mat.copy()
        for i in range(0,len(freq_pun)):
            res[i,:] = res[i,:]*freq_pun[i]
        return res
    except:
        return None
    
# normalized with frequencies
def get_transition_mat(norm_mat,freq_pun):
    try:
        res = norm_mat.copy().reshape((10,10))
        for i in range(0,len(freq_pun)):
            if freq_pun[i]>0:
                res[i,:] = res[i,:]/freq_pun[i]
        return res
    except:
        return None

def update_mat_nb_words(tot,mat,char1,char2,count):
    ind1 = punctuation_vector.index(char1)
    ind2 = punctuation_vector.index(char2)
    s = tot[ind1,ind2] = tot[ind1,ind2]+1
    mat[ind1,ind2] = (mat[ind1,ind2]*(s-1)+count)/s

#function to compute a matrix with the mean of
# words between two punctuation marks:
def mat_nb_words_pun(seq_nb_words_pun):
    try:
        mat_nb_word_pun = np.zeros((NB_sign, NB_sign), dtype='f')
        count_pun =  np.zeros((NB_sign, NB_sign), dtype='f')
        for i in range(1,len(seq_nb_words_pun)-2, 2):
            update_mat_nb_words(count_pun,mat_nb_word_pun,
                seq_nb_words_pun[i],seq_nb_words_pun[i+2],
                seq_nb_words_pun[i+1])
        
        return mat_nb_word_pun
    except:
        return None
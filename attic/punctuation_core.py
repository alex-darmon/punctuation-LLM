#!/usr/bin/env python3
"""
Core punctuation analysis module using the EXACT methods from the paper:
"Pull out all the stops: Textual analysis via punctuation sequences"
Darmon et al. (2020)

This module provides all the functions needed for punctuation stylometry,
matching the paper's implementation exactly.
"""

import numpy as np
import math as ma
from spacy.lang.en import English
from pathlib import Path

# ============================================================================
# CONFIGURATION (from paper's config.py)
# ============================================================================

# Paper's exact punctuation vector (10 marks)
# Note: '^' represents ellipsis '...'
PUNCTUATION_VECTOR = ['!', '"', '(', ')', ',', '.', ':', ';', '?', '^']

# Quotes to normalize to '"'
PUNCTUATION_QUOTES = ["'", """, """, "'", "'"]

# Characters that count as words
ALPHA = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"


# ============================================================================
# PARSING FUNCTIONS (from paper's punctuation_parser.py)
# ============================================================================

def get_tokens_word_nb_punctuation(tokens):
    """
    Extract punctuation sequence with word counts between marks.
    Returns: [word_count, punc, word_count, punc, ...]
    """
    try:
        tokens_word_nb_punctuation = []
        count = 0
        for token in tokens:
            token = str(token)
            # Handle ellipsis
            if token == '...':
                token = '^'
            if token in PUNCTUATION_VECTOR + PUNCTUATION_QUOTES:
                if token in PUNCTUATION_QUOTES:
                    token = '"'
                tokens_word_nb_punctuation += [count, token]
                count = 0
            elif len(set(ALPHA).intersection(set(str(token)))) > 0:
                count += 1
        return tokens_word_nb_punctuation
    except:
        return None


def get_textinfo(text):
    """Parse text and extract punctuation sequence with word counts."""
    if text is None:
        return None
    parser = English()
    parser.max_length = len(text) + 1
    tokens = parser(text)
    return get_tokens_word_nb_punctuation(tokens)


def seq_pun_only(seq):
    """Extract only punctuation marks from the sequence."""
    try:
        if type(seq[0]) == int:
            res = seq[1::2]
        else:
            res = seq[0:-1:2]
        return res
    except:
        return None


def seq_nb_only(seq):
    """Extract only word counts from the sequence."""
    try:
        if type(seq[0]) == int:
            res = seq[0:-1:2]
        else:
            res = seq[1::2]
        return res
    except:
        return None


# ============================================================================
# FEATURE COMPUTATION (from paper's matrix_operations.py)
# ============================================================================

def get_frequencies(tokens, vector=PUNCTUATION_VECTOR):
    """
    Compute f1: frequency distribution of punctuation marks.
    Returns normalized frequencies (sums to 1).
    """
    try:
        freqs = []
        for elt in vector:
            freqs.append(tokens.count(elt))
        total = sum(freqs)
        if total == 0:
            return None
        return list(map(lambda x: x / total, freqs))
    except:
        return None


def _update_mat_tot(tot, mat, char1, char2):
    """Helper function for transition matrix computation."""
    ind1 = PUNCTUATION_VECTOR.index(char1)
    ind2 = PUNCTUATION_VECTOR.index(char2)
    s = tot[ind1] = tot[ind1] + 1
    mat[ind1, :] = mat[ind1, :] * (s - 1)
    mat[ind1, ind2] = mat[ind1, ind2] + 1
    mat[ind1, :] = mat[ind1, :] / s


def transition_mat(seq_pun):
    """
    Compute f2: conditional probability transition matrix.
    P[i,j] = P(next=j | current=i)
    Each row sums to 1.
    """
    try:
        trans_mat = np.zeros((len(PUNCTUATION_VECTOR), len(PUNCTUATION_VECTOR)), dtype='f')
        count_pun = np.zeros(len(PUNCTUATION_VECTOR), dtype='f')
        for i in range(0, len(seq_pun) - 1):
            _update_mat_tot(count_pun, trans_mat, seq_pun[i], seq_pun[i + 1])
        return trans_mat
    except:
        return None


def normalised_transition_mat(mat, freq_pun):
    """
    Compute f3: joint probability transition matrix.
    P_tilde[i,j] = P(current=i AND next=j) = P[i,j] * freq[i]
    Matrix sums to 1.
    """
    try:
        res = mat.copy()
        for i in range(0, len(freq_pun)):
            res[i, :] = res[i, :] * freq_pun[i]
        return res
    except:
        return None


def compute_all_features(punctuation_seq):
    """
    Compute all features (f1, f2, f3) from a punctuation sequence.
    Returns dict with 'f1' (frequency), 'f2' (conditional), 'f3' (joint).
    """
    f1 = get_frequencies(punctuation_seq)
    if f1 is None:
        return None
    
    f2 = transition_mat(punctuation_seq)
    f3 = normalised_transition_mat(f2, f1)
    
    return {
        'f1': f1,
        'f2': f2,
        'f3': f3
    }


# ============================================================================
# DISTANCE FUNCTIONS (from paper's distances.py)
# ============================================================================

def fit_freq_mod2(freq1, freq2):
    """
    Handle zeros in frequency vectors for KL divergence.
    Redistributes probability mass when one vector has zeros.
    """
    if len(freq1) != len(freq2):
        raise Exception("Vectors of different size")
    
    n = len(freq1)
    new_freq1 = [freq1[i] for i in range(n)]
    new_freq2 = [freq2[i] for i in range(n)]

    for j in range(n):
        if new_freq1[j] == 0.0:
            q = new_freq2[j]
            if q != 1.0:
                new_freq2 = list(map(lambda x: x / (1.0 - q), new_freq2))
            new_freq2[j] = 0.0
            
        if new_freq2[j] == 0.0:
            q = new_freq1[j]
            if q != 1.0:
                new_freq1 = list(map(lambda x: x / (1.0 - q), new_freq1))
            new_freq1[j] = 0.0
            
    return (new_freq1, new_freq2)


def d_KL(freq1, freq2):
    """
    Compute KL divergence D_KL(freq1 || freq2) for frequency vectors.
    Uses fit_freq_mod2 to handle zeros.
    """
    res = 0
    try:
        freq1, freq2 = fit_freq_mod2(freq1, freq2)
    except:
        return 0
    
    if freq1 is not None and freq2 is not None:
        for i in range(len(freq1)):
            p = freq1[i]
            q = freq2[i]
            if p * q != 0:
                res += p * ma.log(p / q)
    return res


def d_KL_mat(mat1, mat2):
    """
    Compute KL divergence for matrices (f3).
    D_KL(mat1 || mat2) = sum_{i,j} mat1[i,j] * log(mat1[i,j] / mat2[i,j])
    """
    res = 0
    for i in range(len(PUNCTUATION_VECTOR)):
        for j in range(len(PUNCTUATION_VECTOR)):
            pij = mat1[i, j]
            qij = mat2[i, j]
            if pij * qij != 0:
                res += pij * ma.log(pij / qij)
    return res


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_text(filepath):
    """Load text from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def extract_punctuation(text):
    """Extract punctuation sequence from text."""
    text_info = get_textinfo(text)
    return seq_pun_only(text_info)


def bootstrap_sample(punctuation_seq, sample_size):
    """
    Create a bootstrap sample of given size from the punctuation sequence.
    Returns sorted indices to preserve sequence order.
    """
    if len(punctuation_seq) < sample_size:
        return None
    indices = np.random.choice(len(punctuation_seq), size=sample_size, replace=False)
    indices = np.sort(indices)
    return [punctuation_seq[i] for i in indices]


def split_into_chunks(punctuation_seq, chunk_size):
    """Split sequence into non-overlapping chunks."""
    chunks = []
    for i in range(0, len(punctuation_seq) - chunk_size + 1, chunk_size):
        chunks.append(punctuation_seq[i:i + chunk_size])
    return chunks


# ============================================================================
# HIGH-LEVEL ANALYSIS FUNCTIONS
# ============================================================================

def compare_texts(text1, text2):
    """
    Compare two texts and return KL divergence for f1 and f3.
    """
    seq1 = extract_punctuation(text1)
    seq2 = extract_punctuation(text2)
    
    features1 = compute_all_features(seq1)
    features2 = compute_all_features(seq2)
    
    if features1 is None or features2 is None:
        return None
    
    f1_kl = d_KL(features1['f1'], features2['f1'])
    f3_kl = d_KL_mat(features1['f3'], features2['f3'])
    
    return {
        'f1_kl': f1_kl,
        'f3_kl': f3_kl,
        'seq1_length': len(seq1),
        'seq2_length': len(seq2)
    }


def analyze_text(text):
    """
    Analyze a single text and return all features.
    """
    seq = extract_punctuation(text)
    if seq is None:
        return None
    
    features = compute_all_features(seq)
    if features is None:
        return None
    
    features['punctuation_count'] = len(seq)
    features['punctuation_seq'] = seq
    
    return features


# ============================================================================
# REFERENCE VALUES FROM PAPER
# ============================================================================

PAPER_REFERENCE = {
    'same_author_different_books': {
        'f1_mean': 0.0828,
        'f3_mean': 0.167
    },
    'different_authors': {
        'f1_mean': 0.24,
        'f3_mean': 0.43
    },
    'description': 'Values from Darmon et al. (2020) Figure 9'
}


if __name__ == "__main__":
    # Quick test
    print("Punctuation Core Module")
    print("=" * 50)
    print(f"Punctuation vector: {PUNCTUATION_VECTOR}")
    print(f"Number of punctuation marks: {len(PUNCTUATION_VECTOR)}")
    print("\nPaper reference values:")
    print(f"  Same author, diff books: f1={PAPER_REFERENCE['same_author_different_books']['f1_mean']}, f3={PAPER_REFERENCE['same_author_different_books']['f3_mean']}")
    print(f"  Different authors: f1={PAPER_REFERENCE['different_authors']['f1_mean']}, f3={PAPER_REFERENCE['different_authors']['f3_mean']}")

import pandas as pd
from .pun_modules.tools import (
    empirical_nb_words,
    empirical_nb_sentences,
    punctuation_vector,
    get_tokens_word_nb_punctuation,
    English,
    get_frequencies,
    seq_nb_only,
    seq_pun_only,
    get_tokens_sentences_nb,
    mat_nb_words_pun,
    transition_mat,
    normalised_transition_mat,
    freq_pun_col,
    freq_nb_words_col,
    freq_length_sen_with_col,
    transition_mat_col,
    norm_transition_mat_col,
    mat_nb_words_pun_col
    )
from sklearn.preprocessing import StandardScaler
import pickle
from json import dumps

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

scaler_name = 'scaler.sav'
scaler = pickle.load(open(scaler_name, 'rb'))

def get_features_from_text(text):
  parser = English()
  tokens = parser(text)
  seq_nb_words = get_tokens_word_nb_punctuation(tokens)
  seq_pun =  seq_pun_only(seq_nb_words)

  freq_pun = get_frequencies(seq_pun)
  seq_word_nb_only = seq_nb_only(seq_nb_words)

  freq_word_nb_punctuation = get_frequencies(seq_word_nb_only, range(0, empirical_nb_words))
  seq_length_sen = get_tokens_sentences_nb(seq_nb_words)


  freq_length_sen = get_frequencies(seq_length_sen, range(0, empirical_nb_sentences))

  tran_mat = transition_mat(seq_pun)

  normalised_tran_mat = normalised_transition_mat(tran_mat, freq_pun)

  mat_nb_words = mat_nb_words_pun(seq_nb_words)

  freq_pun_str = pd.DataFrame(list(
       zip([x.replace('"',"''") for x in punctuation_vector][:-1]+['...'], freq_pun)),
   columns=['label','value']).to_dict(orient='records')

  return (seq_pun,
          freq_pun,
          freq_word_nb_punctuation,
          freq_length_sen,
          tran_mat,
          normalised_tran_mat,
          mat_nb_words,
          freq_pun_str)

def get_author_from_model(text, feature_set=None):

    features = freq_pun_col+freq_nb_words_col+freq_length_sen_with_col+\
                 transition_mat_col+ norm_transition_mat_col+mat_nb_words_pun_col

    if feature_set is None:
        (seq_pun,
         freq_pun,
         freq_word_nb_punctuation,
         freq_length_sen,
         tran_mat,
         normalised_tran_mat,
         mat_nb_words,
         _) = get_features_from_text(text)
    else:
        (seq_pun,
         freq_pun,
         freq_word_nb_punctuation,
         freq_length_sen,
         tran_mat,
         normalised_tran_mat,
         mat_nb_words,
         _) = feature_set


    df_res_final = pd.DataFrame(None)

    df_res_final['freq_pun'] = [freq_pun]
    df_res_final['freq_word_nb_punctuation'] = [freq_word_nb_punctuation]
    df_res_final['freq_length_sen'] = [freq_length_sen]
    df_res_final['tran_mat'] = [tran_mat]
    df_res_final['normalised_tran_mat'] = [normalised_tran_mat]
    df_res_final['mat_nb_words'] = [mat_nb_words]


    df_res_final['tran_mat'] = \
    df_res_final['tran_mat'].apply(lambda x:x.flatten())
    df_res_final['normalised_tran_mat'] = \
        df_res_final['normalised_tran_mat'].apply(lambda x:x.flatten())
    df_res_final['mat_nb_words'] = \
        df_res_final['mat_nb_words'].apply(lambda x:x.flatten())


    df_res_final[freq_pun_col] = \
        pd.DataFrame(df_res_final.freq_pun.values.tolist(),
                     index= df_res_final.index)

    df_res_final[freq_nb_words_col] = \
        pd.DataFrame(df_res_final.freq_word_nb_punctuation.values.tolist(),
                     index= df_res_final.index)

    df_res_final[freq_length_sen_with_col] = \
        pd.DataFrame(df_res_final.freq_length_sen.values.tolist(),
                     index= df_res_final.index)

    df_res_final[transition_mat_col] = \
        pd.DataFrame(df_res_final.tran_mat.values.tolist(),
                     index= df_res_final.index)

    df_res_final[norm_transition_mat_col] = \
        pd.DataFrame(df_res_final.normalised_tran_mat.values.tolist(),
                     index= df_res_final.index)

    df_res_final[mat_nb_words_pun_col] = \
        pd.DataFrame(df_res_final.mat_nb_words.values.tolist(),
                     index= df_res_final.index)
    df_res_final = df_res_final[features]
    df_res_final = scaler.transform(df_res_final)

    prediction = loaded_model.predict(df_res_final)
    return prediction[0]

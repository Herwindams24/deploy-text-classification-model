import os
os.environ['REQUESTS_CA_BUNDLE'] = "C:/Users/ASUS/Downloads/AmazonRootCA1.crt"
os.environ["WANDB_API_KEY"] = "0" ## to silence warning
# os.environ["CURL_CA_BUNDLE"]=""
# os.environ["REQUEST_CA_BUNDLE"]=""

import numpy as np
import pandas as pd
import re
import tensorflow as tf
import h5py
import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, BertModel, BertConfig, TFBertModel
from flask import Flask, request, render_template

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy() # for CPU and single GPU
    print('Number of replicas:', strategy.num_replicas_in_sync)

app = Flask(__name__, template_folder='templateFiles', static_folder='staticFiles')

########################################
# Preprocessing Text
## Case folding
def lowercase(text):
    return text.lower()

## Cleaning the corpus
def clean_text(text):
    text = re.sub('\n',' ',text) # Remove every '\n'
    text = re.sub('rt',' ',text) # Remove every retweet symbol
    text = re.sub('user',' ',text) # Remove every username
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text) # Remove every URL
    text = re.sub('  +', ' ', text) # Remove extra spaces
    text = re.sub('[^0-9a-zA-Z]+', ' ', text) 
    text = re.sub('url', ' ', text) 
    return text

## Text Normalization
alay_dict = pd.read_csv('/Python-Project/github.com/Herwindams24/deploy-text-classification-model-using-Flask/WebsiteApp/input/new_kamusalay.csv', encoding='latin-1', header=None)
alay_dict = alay_dict.rename(columns={0: 'original', 1: 'replacement'})
alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))
def normalize_alay(text):
    return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])

## Stop words removal
id_stopword_dict = pd.read_csv('/Python-Project/github.com/Herwindams24/deploy-text-classification-model-using-Flask/WebsiteApp/input/stopwordbahasa.csv', header=None)
id_stopword_dict = id_stopword_dict.rename(columns={0: 'stopword'})
def remove_stopword(text):
    text = ' '.join(['' if word in id_stopword_dict.stopword.values else word for word in text.split(' ')])
    text = re.sub('  +', ' ', text) # Remove extra spaces
    text = text.strip()
    return text

def preprocess(text):
    text = lowercase(text) # 1
    text = clean_text(text) # 2
    text = normalize_alay(text) # 3
    text = remove_stopword(text) # 4
    return text
########################################

########################################
# Tokenizing
bert_base='cahya/bert-base-indonesian-522M'
tokenizer = BertTokenizer.from_pretrained(bert_base)

def bert_encode(data, maximum_length = 52) :
    input_ids = []
    tokens = tokenizer.encode_plus(data, 
            add_special_tokens=True,
            max_length=maximum_length,
            pad_to_max_length=True,
            return_attention_mask=False,
            return_token_type_ids=False,
            truncation=True)
    input_ids.append(tokens['input_ids'])
    return np.array(input_ids)
########################################

########################################
# Load Model
path = '/Python-Project/github.com/Herwindams24/deploy-text-classification-model-using-Flask/WebsiteApp/input/model_tf.h5'
with strategy.scope():
    model_tf = load_model(path)
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model_tf.compile(loss='binary_crossentropy',optimizer=adam_optimizer,metrics=['accuracy'])
########################################
def preprocessing(text):
    review_text = preprocess(text)
    return review_text

def encode(review_text):
    encoded_text = bert_encode(review_text)
    return encoded_text
########################################
def prediction(encoded_text):

    # Predict Text
    predictions = model_tf.predict(encoded_text, verbose=1)
    final_predictions = tf.cast(tf.round(predictions), tf.int32).numpy().flatten()

    return final_predictions
########################################

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    to_predict_list = request.form.to_dict()
    review_text = preprocessing(to_predict_list['tweet'])
    encoded_text = encode(review_text)
    final_predictions = prediction(encoded_text)
    
    newarr = np.array_split(final_predictions, 12)

    hateSpeech = newarr[0]
    abusive = newarr[1]
    hs_Individual =newarr[2]
    hs_Group = newarr[3]
    hs_Religion = newarr[4]
    hs_Race = newarr[5]
    hs_Physical = newarr[6]
    hs_Gender = newarr[7]
    hs_Other = newarr[8]
    hs_Weak	= newarr[9]
    hs_Moderate = newarr[10]
    hs_Strong = newarr[11]


    # Render
    return render_template('predict.html', tweet_text = to_predict_list['tweet'], cleaned_text = review_text, 
                       hateSpeech = hateSpeech,  abusive = abusive, 
                       hs_Individual = hs_Individual, hs_Group = hs_Group, 
                       hs_Religion = hs_Religion, hs_Race = hs_Race, hs_Physical = hs_Physical, hs_Gender = hs_Gender, hs_Other = hs_Other,
                       hs_Weak = hs_Weak, hs_Strong = hs_Strong, hs_Moderate = hs_Moderate)

if __name__ == "__main__":
    app.run()
    #app.run()
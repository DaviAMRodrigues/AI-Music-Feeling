from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from pytube import YouTube
from dotenv import load_dotenv, find_dotenv

# Função para extrair características do áudio
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

# Carregar dados e treinar o modelo
def train_model():
    df = pd.read_csv('mean_emotion_ratings.csv', sep=',', decimal='.')
    audio_dir = 'test-stimuli-200-2009-05-29'
    audio_files = [os.path.join(audio_dir, file) for file in os.listdir(audio_dir) if file.endswith('.wav')]
    features = [extract_features(file) for file in audio_files]
    feature_columns = [f'mfcc_{i}' for i in range(13)]
    df_audios = pd.DataFrame(features, columns=feature_columns)
    
    df_audios['Num'] = range(1, len(df_audios) + 1)
    df['Num'] = range(1, len(df_audios) + 1)
    
    df_complete = pd.merge(df, df_audios, on='Num')
    df_complete.drop('Num', axis=1, inplace=True)
    df_complete.set_index('Nro', inplace=True)
    
    X_df = ['Scary', 'Happy', 'Sad', 'Peaceful']
    X = df_complete.loc[:, ~df_complete.columns.isin(X_df)]
    Y = df_complete[X_df]
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    
    rt_reg = LinearRegression()
    rt_reg.fit(x_train, y_train)
    
    scores = cross_val_score(rt_reg, x_test, y_test, scoring='neg_mean_squared_error', cv=10)
    rt_rmse_scores = np.sqrt(-scores)
    
    return rt_reg

# Inicializa o modelo
model = train_model()

# Criação da aplicação Flask
app = Flask(__name__)

# Endpoint para prever emoções de um áudio
@app.route("/predict-emotion", methods=['POST'])
def predict_emotion():
    data = request.get_json()
    link = data.get('link')
    
    if not link:
        return jsonify({"error": "Link do YouTube é necessário"}), 400
    
    yt = YouTube(link)
    ytaudio = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
    filename = 'audio.wav'
    ytaudio.download(filename=filename)
    
    features = [extract_features(filename)]
    feature_columns = [f'mfcc_{i}' for i in range(13)]
    df_usuario = pd.DataFrame(features, columns=feature_columns)
    
    prediction = model.predict(df_usuario)
    
    return jsonify({"Prediction in SHSP (Scary, Happy, Sad & Peaceful)": prediction.tolist()}), 200

if __name__ == '__main__':
    app.run(debug=True)

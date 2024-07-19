from flask import Flask, request, render_template, jsonify 
import yt_dlp 
import sys 
from pydub  import AudioSegment
import os
from pytube import YouTube

app =Flask(__name__) 
DOWNLOAD_FOLDER = 'downloads' 
if not os.path.exists[DOWNLOAD_FOLDER]:os.makedirs[DOWNLOAD_FOLDER]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def convert():
    url = request.form(url)
    yt = YouTube(url)
    audio_stream = yt.stream.filter(only_audio=True).first()
    audio_file = audio_stream.convert(out_path="DOWNLOAD_FOLDER")
        
    base, ext = os.path.splitext(audio_file)
    mp3_file = base + '.mp3'
    
    AudioSegment.from_file(audio_file).export(mp3_file, format="mp3")
    os.remove(audio_file)
    filename = os.path.basename(mp3_file)
    return jsonify (filename=filename)

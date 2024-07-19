from flask import Blueprint, request, render_template, app, jsonify
import yt_dlp 
from pydub import AudioSegment  
import os
import sys

main = Blueprint('main',__name__)

@main.route('/')
def home():
 return render_template('index.html')
        
@main.route('/download', methods=['POST'])
def download(ydl_opts):
      data= request.get_json()
      url=data.get('url')
      if not url:
           return jsonify({'error':'Url is required'}), 400

      output_dir='downloads'
      if not os.path.exists(output_dir):
            os.mkdir(output_dir)
 
      ydl_opts = {
       'format':'bestaudio/best',
       'outtmpl':f'(output_dir)/%(title)s.%(ext)s',
       'postproccessors':[{
       'key':'FFmpegExtractAudio', 
       'preferredcodec':'mp3', 
       'preferredquality':'192',
      }],
  }
      try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
               info_dict=ydl.extract_info(urldownload=True)
               file_path=ydl.prepare_filename(info_dict).replace('.m4a','.mp3')
            return jsonify({'message':'download successful','file':file_path}), 200
      except Exception as e:
            return jsonify({'error': str(e)}), 500
 
if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0')
       
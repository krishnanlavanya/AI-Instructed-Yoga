from flask import Flask, render_template, request, redirect, url_for, abort
from helpers import extract_poses, save_angles, read_poses_json
import time
import os
import shutil

if not os.path.exists('uploads'):
    os.mkdir('uploads')
if not os.path.exists('templates/processed_cache'):
    os.mkdir('templates/processed_cache')

app = Flask(__name__, static_folder='./templates/')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():

    print(request.form)
    if request.method == 'POST':
        print("Logging in ")
        username = request.form['email']
        password = request.form['password']
        # check if the username and password are correct
        if username == 'admin@admin.com' and password == 'admin':
            return redirect(url_for('upload_image'))
        else:
            error = 'Invalid Credentials. Please try again.'
            return render_template('index.html', error=error)
    else:
        abort(403)

@app.route('/upload_image', methods=['POST', 'GET'])
def upload_image():
    poses, _ = read_poses_json()
    # poses = enumerate(poses)
    if request.method == 'POST':
        image = request.files['image']
        if image:
            image.save('./uploads/latest.jpg')
            angles = extract_poses('./uploads/latest.jpg')

        return render_template('upload_screen.html', image='processed_cache/1.jpg', 
                                                           angles=angles, 
                                                           poses=poses)
    else:
        return render_template('upload_screen.html', poses=poses)

@app.route('/confirm_angles', methods=['POST', 'GET'])
def confirm_angles():
    if request.method == 'POST':
        angles = eval(request.form['angles'])
        pose_name = request.form['pose_name']
        save_angles(angles, pose_name)
        shutil.copyfile('./uploads/latest.jpg', f'./pose_images/{pose_name}.jpg')

        return redirect(url_for('upload_image'))
    else:
        abort(403)

if __name__ == '__main__':
    app.run(debug=True)
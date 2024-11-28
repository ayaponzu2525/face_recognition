import pickle  # 特徴量を保存するために使います
import sys

sys.path.append(r"c:\users\ayapo\.pyenv\pyenv-win\versions\3.10.0\lib\site-packages")
predictor_path = r"c:\Users\ayapo\.pyenv\pyenv-win\versions\3.10.0\Lib\site-packages\face_recognition_models\models\shape_predictor_68_face_landmarks.dat"
import face_recognition

# 学習用顔画像のファイル名をリストに格納
train_img_names = ["test1_ayaponzu.jpg", "test2_ayaponzu.jpg"]

train_img_encodings = []

# 学習データの顔画像を読み込み、特徴量を取得
for name in train_img_names:
    # 画像を読み込む
    img = face_recognition.load_image_file(name)

    # 顔の特徴量（エンコード）を取得
    encoding = face_recognition.face_encodings(img)

    # 顔が検出された場合、特徴量を保存
    if encoding:
        train_img_encodings.append(encoding[0])  # 最初の顔のみ取り出す

# 特徴量をpickleで保存する
with open('train_img_encodings.pkl', 'wb') as f:
    pickle.dump(train_img_encodings, f)

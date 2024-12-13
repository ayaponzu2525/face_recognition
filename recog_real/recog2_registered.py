import numpy as np
import pickle
import cv2
import sys
import os

sys.path.append(r"c:\users\ayapo\.pyenv\pyenv-win\versions\3.10.0\lib\site-packages")
import face_recognition

# データフォルダのパス
data_dir = r'C:\Users\ayapo\Documents\hightech_local\face_recognition\face_recognition\register_data'

# 複数の人物の特徴量と名前を読み込む
train_img_encodings = []
train_img_names = []

# データフォルダ内の全ての*_encodings.pklファイルを処理
for filename in os.listdir(data_dir):
    if filename.endswith('_encodings.pkl'):
        # 対応する名前ファイルを探す
        name_filename = filename.replace('_encodings.pkl', '_names.pkl')
        
        # エンコーディングの読み込み
        with open(os.path.join(data_dir, filename), 'rb') as f_enc:
            encodings = pickle.load(f_enc)
        
        # 名前の読み込み
        with open(os.path.join(data_dir, name_filename), 'rb') as f_names:
            names = pickle.load(f_names)
        
        # リストに追加
        train_img_encodings.extend(encodings)
        train_img_names.extend(names)

# NumPy配列に変換
train_img_encodings = np.array([list(encoding) for encoding in train_img_encodings])

# カメラを開く
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("カメラの映像取得に失敗しました")
        break

    # 顔の検出
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # 学習済み特徴量との比較
    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        dists = face_recognition.face_distance(train_img_encodings, face_encoding)

        # 最も近い距離を取って一致判定
        best_match_index = np.argmin(dists)
        if min(dists) < 0.40:  # 距離が0.40未満なら一致と判定
            label = train_img_names[best_match_index]  # 一致した名前を取得
            color = (0, 255, 0)  # 緑色で囲む
        else:
            label = "Unknown"  # 一致しなかった場合
            color = (0, 0, 255)  # 赤色で囲む

        # 顔を□で囲む
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        # 顔の上に名前または「Unknown」を表示
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # 結果をウィンドウに表示
    cv2.imshow("Face Recognition", frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラを解放してウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
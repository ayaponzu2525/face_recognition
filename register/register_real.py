import cv2
import sys

sys.path.append(r"c:\users\ayapo\.pyenv\pyenv-win\versions\3.10.0\lib\site-packages")
predictor_path = r"c:\Users\ayapo\.pyenv\pyenv-win\versions\3.10.0\Lib\site-packages\face_recognition_models\models\shape_predictor_68_face_landmarks.dat"
import face_recognition
import pickle

# 名前と顔画像の特徴量を保存するためのリスト
train_img_encodings = []
train_img_names = []

# カメラを開く
cap = cv2.VideoCapture(0)

# ユーザーに名前を入力してもらう
name = input("登録する名前を入力してください: ")

while True:
    ret, frame = cap.read()
    if not ret:
        print("カメラの映像取得に失敗しました")
        break

    # 顔の検出
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # 顔の特徴量をリストに追加
    for face_encoding in face_encodings:
        train_img_encodings.append(face_encoding)
        train_img_names.append(name)

        # 顔の周りに四角を描く
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # 顔画像を表示
    cv2.imshow("Face Registration", frame)

    # 'q'キーで登録終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 登録した特徴量と名前を保存する
with open('train_img_encodings.pkl', 'wb') as f_enc:
    pickle.dump(train_img_encodings, f_enc)

with open('train_img_names.pkl', 'wb') as f_names:
    pickle.dump(train_img_names, f_names)

# カメラを解放してウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()

print("登録が完了しました。")

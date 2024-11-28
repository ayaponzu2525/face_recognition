import pickle
import cv2
import sys

sys.path.append(r"c:\users\ayapo\.pyenv\pyenv-win\versions\3.10.0\lib\site-packages")
predictor_path = r"c:\Users\ayapo\.pyenv\pyenv-win\versions\3.10.0\Lib\site-packages\face_recognition_models\models\shape_predictor_68_face_landmarks.dat"
import face_recognition

# 保存された特徴量と名前を読み込む
with open(r'..\register\train_img_encodings.pkl', 'rb') as f_enc:
    train_img_encodings = pickle.load(f_enc)

with open(r'..\register\train_img_names.pkl', 'rb') as f_names:
    train_img_names = pickle.load(f_names)

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
        best_match_index = dists.argmin()
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

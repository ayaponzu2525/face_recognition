import pickle
import cv2
# 保存された特徴量を読み込む
with open(r'..\register\train_img_encodings.pkl', 'rb') as f:
    # pickleを使ってファイルを読み込む
    train_img_encodings = pickle.load(f)

# これで、train_img_encodingsには保存した特徴量がリスト形式で読み込まれています
print("特徴量読み込めた")

import sys

sys.path.append(r"c:\users\ayapo\.pyenv\pyenv-win\versions\3.10.0\lib\site-packages")
predictor_path = r"c:\Users\ayapo\.pyenv\pyenv-win\versions\3.10.0\Lib\site-packages\face_recognition_models\models\shape_predictor_68_face_landmarks.dat"
import face_recognition

# カメラを開く
cap = cv2.VideoCapture(0)

# カメラが開けなかった場合
if not cap.isOpened():
    print("Error: Could not open camera.")
    sys.exit()

while True:
    # カメラから顔を検出して特徴量を計算
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # 顔の検出
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # 学習済み特徴量との比較
    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        # 顔の特徴量と保存した特徴量との距離を計算
        dists = face_recognition.face_distance(train_img_encodings, face_encoding)

        # 最も近い距離を取って一致判定
        if min(dists) < 0.40:  # 距離が0.40未満なら一致と判定
            label = "Match"
            color = (0, 255, 0)  # 緑色で囲む
        else:
            label = "No Match"
            color = (0, 0, 255)  # 赤色で囲む

        # 顔を□で囲む
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        # 顔の上にラベルを表示
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # 結果をウィンドウに表示
    cv2.imshow("Face Recognition", frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラを解放してウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
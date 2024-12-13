import cv2
import sys

sys.path.append(r"c:\users\ayapo\.pyenv\pyenv-win\versions\3.10.0\lib\site-packages")
predictor_path = r"c:\Users\ayapo\.pyenv\pyenv-win\versions\3.10.0\Lib\site-packages\face_recognition_models\models\shape_predictor_68_face_landmarks.dat"
import face_recognition
import pickle
import os



# 保存するファイル名のディレクトリ
register_dir = '../register_data'

# 登録された顔特徴量と名前を保存するリスト
all_train_img_encodings = []
all_train_img_names = []

# ディレクトリが存在しない場合、作成
if not os.path.exists(register_dir):
    os.makedirs(register_dir)

while True:
    # ユーザーに名前を入力してもらう
    name = input("登録する名前を入力してください (終了するには 'q' を入力): ")

    if name.lower() == 'q':
        print("登録を終了します。")
        break

    # 特徴量のリスト
    train_img_encodings = []
    train_img_names = []

    print(f"{name}さんの顔を登録します...")
    
    # カメラを開く
    cap = cv2.VideoCapture(0)

    # 顔認識フレーム数の設定（例：5フレーム顔が検出されるまで）
    recognized_faces = 0
    target_frames = 15

    while recognized_faces < target_frames:
        ret, frame = cap.read()
        if not ret:
            print("カメラの映像取得に失敗しました")
            break

        # 顔の検出
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            # 顔の特徴量をリストに追加
            train_img_encodings.append(face_encoding)
            train_img_names.append(name)
            recognized_faces += 1

            # 顔の周りに四角を描く
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # 顔が検出されていれば画面に表示
        cv2.imshow(f"登録中 - {name}", frame)

        # 'q'キーで登録を終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 一定フレーム数顔を認識したら保存
    if recognized_faces >= target_frames:
        # 名前ごとに特徴量を保存
        encoding_file = os.path.join(register_dir, f"{name}_encodings.pkl")
        with open(encoding_file, 'wb') as f_enc:
            pickle.dump(train_img_encodings, f_enc)

        # 名前も一緒に保存
        name_file = os.path.join(register_dir, f"{name}_names.pkl")
        with open(name_file, 'wb') as f_names:
            pickle.dump(train_img_names, f_names)

        print(f"{name}さんの顔登録が完了しました。")
        
            # カメラを解放してウィンドウを閉じる
    cap.release()
    cv2.destroyAllWindows()

    # 続けて登録するか終了するか
    continue_registration = input("次の名前を登録しますか？ (y/n): ")
    if continue_registration.lower() != 'y':
        print("全ての顔登録が完了しました。")
        break
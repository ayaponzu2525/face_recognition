import sys

sys.path.append(r"c:\users\ayapo\.pyenv\pyenv-win\versions\3.10.0\lib\site-packages")
predictor_path = r"c:\Users\ayapo\.pyenv\pyenv-win\versions\3.10.0\Lib\site-packages\face_recognition_models\models\shape_predictor_68_face_landmarks.dat"
import face_recognition
import matplotlib.pyplot as plt
import dlib

# モデルパスを指定
predictor_path = r"c:\Users\ayapo\.pyenv\pyenv-win\versions\3.10.0\Lib\site-packages\face_recognition_models\models\shape_predictor_68_face_landmarks.dat"

# 学習データの画像ファイル名をリストに格納
train_img_names = ["train1.jpg", "train2.jpg", "train3.jpg","train4.jpg","train5.jpg","train6.jpg","train7.jpg"]  # 複数の学習画像
# テストデータの画像ファイル名をリストに格納
test_img_names = ["test1.jpg", "test2.jpg", "test3.jpg","abe1.jpg","abe2.jpg","oji1.jpg","oji2.jpg"]  # 複数のテスト画像

# dlibのshape_predictorをロード
try:
    predictor = dlib.shape_predictor(predictor_path)
    print(f"Model loaded successfully from {predictor_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# 学習データの顔画像を読み込み、特徴量を抽出
train_img_encodings = []
train_img_files = []

for name in train_img_names:
    try:
        img = face_recognition.load_image_file(name)
        locations = face_recognition.face_locations(img, model='hog')
        assert len(locations) == 1, f"Error: {name}に複数の顔または顔が検出されませんでした。"
        (encoding,) = face_recognition.face_encodings(img, locations)
        train_img_encodings.append(encoding)
        train_img_files.append(name)  # 画像名を保存
    except Exception as e:
        print(f"Error processing {name}: {e}")

# テストデータの顔画像を読み込み、認証処理を行う
for test_name in test_img_names:
    try:
        test_img = face_recognition.load_image_file(test_name)
        test_locations = face_recognition.face_locations(test_img, model='hog')
        assert len(test_locations) == 1, f"Error: {test_name}に複数の顔または顔が検出されませんでした。"
        (test_encoding,) = face_recognition.face_encodings(test_img, test_locations)

        # 学習データとテストデータの特徴量を比較
        dists = face_recognition.face_distance(train_img_encodings, test_encoding)

        # 距離が閾値以下であれば一致と判定
        results = []
        for train_name, dist in zip(train_img_files, dists):
            match = dist < 0.40
            results.append((train_name, dist, match))

        # 結果を出力
        print(f"Results for {test_name}:")
        for train_name, dist, match in results:
            print(f" - Compared to {train_name}: Distance = {dist:.2f}, Match = {match}")
    except Exception as e:
        print(f"Error processing {test_name}: {e}")

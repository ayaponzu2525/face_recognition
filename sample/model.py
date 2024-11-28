import os
import sys
sys.path.append(r"c:\users\ayapo\.pyenv\pyenv-win\versions\3.10.0\lib\site-packages")
# 手動でモデルのパスを設定
predictor_path = r"c:\Users\ayapo\.pyenv\pyenv-win\versions\3.10.0\Lib\site-packages\face_recognition_models\models\shape_predictor_68_face_landmarks.dat"
print(f"予測器のパス: {predictor_path}")
print(f"ファイルが存在するか: {os.path.exists(predictor_path)}")

# ファイルが存在する場合にモデルを読み込む
if os.path.exists(predictor_path):
    import dlib
    predictor = dlib.shape_predictor(predictor_path)
    print("モデルが正常に読み込まれました")
else:
    print("ファイルが見つかりません")

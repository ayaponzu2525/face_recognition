import os
import face_recognition_models

# モデルディレクトリを取得
model_dir = face_recognition_models.face_recognition_model_location()
print(f"モデルディレクトリ: {model_dir}")

# 予測器のパスを正しく構築
predictor_path = os.path.join(os.path.dirname(model_dir), "models", "shape_predictor_68_face_landmarks.dat")
print(f"予測器のパス: {predictor_path}")
print(f"ファイルが存在するか: {os.path.exists(predictor_path)}")


# 手動でモデルのパスを設定
predictor_path = r"c:\Users\ayapo\.pyenv\pyenv-win\versions\3.10.0\Lib\site-packages\face_recognition_models\models\shape_predictor_68_face_landmarks.dat"
print(f"予測器のパス: {predictor_path}")
print(f"ファイルが存在するか: {os.path.exists(predictor_path)}")



# モデルディレクトリを取得
model_dir = face_recognition_models.face_recognition_model_location()
print(f"モデルディレクトリ: {model_dir}")

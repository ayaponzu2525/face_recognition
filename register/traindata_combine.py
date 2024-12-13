import pickle
import os

# 特徴量と名前を格納するリスト
train_img_encodings = []
train_img_names = []

folder_path = r"C:\Users\ayapo\Documents\hightech_local\face_recognition\face_recognition\register_data"

# フォルダ内のすべての pkl ファイルを読み込む
for pkl_file in os.listdir(folder_path):
    if pkl_file.endswith(".pkl"):
        with open(os.path.join(folder_path, pkl_file), 'rb') as f:
            data = pickle.load(f)
            # ファイル名から名前を取得 (例: ayaponzu_encoding.pkl -> ayaponzu)
            name = os.path.splitext(pkl_file)[0]
            train_img_encodings.append(data)  # 特徴量を追加
            train_img_names.append(name)  # 名前を追加

print(f"Loaded {len(train_img_encodings)} encodings for {len(set(train_img_names))} unique names.")

# 保存する場合
with open('train_img_encodings.pkl', 'wb') as f_enc:
    pickle.dump(train_img_encodings, f_enc)

with open('train_img_names.pkl', 'wb') as f_names:
    pickle.dump(train_img_names, f_names)

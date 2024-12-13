import pickle
import os

folder_path = r"C:\Users\ayapo\Documents\hightech_local\face_recognition\face_recognition\register_data"

# フォルダ内のすべての pkl ファイルを取得
pkl_files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]

for pkl_file in pkl_files:
    with open(os.path.join(folder_path, pkl_file), 'rb') as f:
        data = pickle.load(f)
        print(f"File: {pkl_file}")
        print(f"Data: {data}\n")

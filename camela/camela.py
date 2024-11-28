import cv2

# カメラのキャプチャを開始
cap = cv2.VideoCapture(0)

while True:
    # フレームを取得
    ret, frame = cap.read()
    
    # フレームが取得できない場合は終了
    if not ret:
        break
    
    # フレームの処理を行う（次のステップ）
    # ここで顔認識を行う
    # ...

    # 画面にフレームを表示
    cv2.imshow("Video", frame)

    # 'q'キーを押すと終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラを解放し、ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()

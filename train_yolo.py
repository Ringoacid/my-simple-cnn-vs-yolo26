from ultralytics import YOLO

def main():
    # モデルの初期化
    model = YOLO("yolo26n.pt")  # 事前学習済みのYOLO26nモデルを使用

    # データセットのパスを指定してトレーニングを開始
    model.train(data="dataset/data.yaml")

if __name__ == "__main__":
    main()
from PIL import Image
from torchvision import transforms

def image_to_tensor(image_path):
    img = Image.open(image_path)
    transform = transforms.ToTensor()
    return transform(img)


# --- 以下、テスト用のコード ---

def main():
    img = image_to_tensor('sample.jpg')
    print(img.shape) # torch.Size([3, 640, 640]) # チャネル・高さ・幅

    print(f"白：{img[:,0,0]}")
    print(f"赤：{img[:,0,1]}")
    print(f"緑：{img[:,0,2]}")
    print(f"青：{img[:,0,3]}")
    print(f"黒：{img[:,0,4]}")
    

if __name__ == "__main__":
    main()
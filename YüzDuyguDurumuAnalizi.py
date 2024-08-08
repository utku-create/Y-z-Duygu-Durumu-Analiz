#import kısmı
import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
from tkinter import Tk, Button, Label, filedialog
from PIL import Image, ImageTk

#fonksiyonlar
def analyze_emotion(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    analysis = DeepFace.analyze(img_rgb, actions=['emotion'], enforce_detection=False)

    if isinstance(analysis, list):
        analysis = analysis[0]

    dominant_emotion = analysis['dominant_emotion']
    print("Duygu Analizi Sonuçları:", dominant_emotion)

    plt.imshow(img_rgb)
    plt.title(f"Duygu: {dominant_emotion}")
    plt.axis('off')
    plt.show()

def open_file():
    file_path = filedialog.askopenfilename(title="Bir resim seçin", filetypes=[("Image files", "*.jpg *.jpeg *.png")])

    if file_path:
        img = Image.open(file_path)
        img.thumbnail((250, 250))
        img = ImageTk.PhotoImage(img)
        preview_label.config(image=img)
        preview_label.image = img
        analyze_emotion(file_path)

def analyze_emotion_live():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera açılmadı")
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            analysis = DeepFace.analyze(img_rgb, actions=['emotion'], enforce_detection=False)

            if isinstance(analysis, list):
                analysis = analysis[0]

            dominant_emotion = analysis['dominant_emotion']
            cv2.putText(frame, f'Duygu: {dominant_emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        except:
            pass

        cv2.imshow('Kamera - Duygu Durumu Analizi', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

#ekran kısmı

root = Tk()
root.title("Duygu Durumu Analizi")
root.geometry("400x400")

select_button = Button(root, text="Fotoğraf Seç", command=open_file)
select_button.pack(pady=20)

camera_button = Button(root, text="Kamera ile Anlık Analiz", command=analyze_emotion_live)
camera_button.pack(pady=20)

preview_label = Label(root)
preview_label.pack()

root.mainloop()

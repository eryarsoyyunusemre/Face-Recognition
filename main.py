import cv2
import face_recognition
import os
import subprocess

# Kişinin fotoğrafını çekmek için kamera oluşturma
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Kaydedilen kişilerin yüz verilerini ve isimlerini tutan listeler
known_faces = []
known_names = []

# Kaydedilen kişilerin isimlerini ve fotoğraflarını yükleme
def load_known_faces():
    folder = "yuzler/"
    images = os.listdir(folder)
    for image in images:
        name = os.path.splitext(image)[0]
        known_names.append(name)
        img_path = os.path.join(folder, image)
        face_image = face_recognition.load_image_file(img_path)
        face_encoding = face_recognition.face_encodings(face_image)[0]
        known_faces.append(face_encoding)

# Kişinin fotoğrafını çekme
def take_photo():
    ret, frame = video_capture.read()
    return frame

# Kaydedilen kişinin yüzünü tanıma ve adını alma
def recognize_person(face_encoding):
    matches = face_recognition.compare_faces(known_faces, face_encoding)
    if True in matches:
        matched_index = matches.index(True)
        person_name = known_names[matched_index]
        return person_name
    return None

# Kaydedilen kişilerin yüz verilerini ve isimlerini yükleme
load_known_faces()

# Sadece bir kez hoş geldiniz demek için flag
welcome_message_played = False

while True:
    # Görüntü yakalama
    ret, frame = video_capture.read()

    # Yüzleri tespit etme
    face_locations = face_recognition.face_locations(frame, model="hog")
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Bulunan yüzleri çerçeveleme ve tanıma
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        person_name = recognize_person(face_encoding)

        # Eşleşme kontrolü
        if person_name is not None and not welcome_message_played:
            # Yüzü çerçeveleme ve ismi yazma (Yeşil renkte)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, person_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            subprocess.run(['say', 'Hoş geldiniz, ' + person_name])  # Seslendirme işlemi
            welcome_message_played = True
        else:
            # Yüzü çerçeveleme ve ismi yazma (Kırmızı renkte)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, person_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            if not welcome_message_played:  # Yalnızca hoş geldiniz mesajı söylendikten sonra yeni kişi sorma
                # Tanınmayan kişinin fotoğrafını çekme
                new_face_image = take_photo()

                # Kişinin ismini alıp fotoğrafını kaydetme
                name = input("Yeni kişinin ismini girin: ")
                img_path = f"yuzler/{name}.jpg"
                cv2.imwrite(img_path, new_face_image)
                print(f"{name} isimli kişi kaydedildi.")

                # Yeni kişinin yüz verilerini ve ismini yükleme
                face_encoding = face_recognition.face_encodings(new_face_image)[0]
                known_faces.append(face_encoding)
                known_names.append(name)

    # Sonuçları gösterme
    cv2.imshow('Video', frame)

    # Çıkış için 'q' tuşuna basılmasını kontrol etme
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()

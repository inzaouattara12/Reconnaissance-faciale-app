import os
import cv2
import pandas as pd
from deepface import DeepFace
import numpy as np
import streamlit as st
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

# Configuration Streamlit
st.title("Système de Reconnaissance Faciale")
st.sidebar.title("Options")
save_path = "faces/"
os.makedirs(save_path, exist_ok=True)

# Section de capture ou upload d'image
name = st.text_input("Entrez votre nom :")

option = st.radio("Choisissez une option :", ("Prendre une photo avec la webcam", "Charger une photo depuis le disque"))

image_path = None

if option == "Prendre une photo avec la webcam":
    # Capture de la webcam
    run_webcam = st.button("Démarrer la webcam")
    
    if run_webcam:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Erreur lors de l'accès à la webcam.")
                break

            # Affichage de la vidéo
            stframe.image(frame, channels="BGR", use_container_width=True)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                image_path = os.path.join(save_path, f"{name}.jpg")
                cv2.imwrite(image_path, frame)
                st.success(f"Image enregistrée sous {image_path}")
                break

        cap.release()

elif option == "Charger une photo depuis le disque":
    # Charger une image
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(title="Sélectionnez une image", filetypes=[("Images", "*.jpg *.jpeg *.png")])
    if image_path:
        st.image(image_path, caption="Image sélectionnée", use_container_width=True)
        st.success(f"Image sélectionnée : {image_path}")
    else:
        st.warning("Aucune image sélectionnée.")

# Traitement de l'image (capture ou chargée)
if image_path and os.path.exists(image_path):
    try:
        embedding = DeepFace.represent(img_path=image_path, model_name="VGG-Face", detector_backend="mtcnn")[0]["embedding"]

        csv_file = "faces.csv"
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame(columns=["name"] + [f"e{i}" for i in range(len(embedding))])

        new_data = pd.DataFrame([[name] + embedding], columns=df.columns)
        df = pd.concat([df, new_data], ignore_index=True)

        df.to_csv(csv_file, index=False)
        st.success(f"Encodage sauvegardé pour {name} dans {csv_file}")

    except Exception as e:
        st.error(f"Erreur lors de l'extraction des embeddings : {e}")
else:
    st.warning("Erreur : Aucune image valide fournie.")

# Reconnaissance faciale en temps réel
csv_file = "faces.csv"
df = pd.read_csv(csv_file)

presence_file = "presence.xlsx"

# Charger les présences déjà enregistrées
if os.path.exists(presence_file):
    presence_df = pd.read_excel(presence_file)
    already_present = set(presence_df["name"].tolist())
else:
    presence_df = pd.DataFrame(columns=["name", "datetime", "present"])
    already_present = set()

# Démarrer la webcam pour reconnaissance en temps réel
stframe = st.empty()

cap = cv2.VideoCapture(0)

if st.button("Commencer la reconnaissance faciale"):
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Erreur lors de l'accès à la webcam.")
            break

        stframe.image(frame, channels="BGR", use_container_width=True)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            test_img_path = "test.jpg"
            cv2.imwrite(test_img_path, frame)
            st.success("Image capturée pour comparaison.")

            try:
                embedding = DeepFace.represent(
                    img_path=test_img_path,
                    model_name="VGG-Face",
                    detector_backend="mtcnn"
                )[0]["embedding"]

                min_distance = float("inf")
                recognized_name = "Inconnu"

                for _, row in df.iterrows():
                    stored_embedding = np.array(row[1:], dtype=np.float64)
                    distance = np.linalg.norm(stored_embedding - embedding)

                    if distance < min_distance:
                        min_distance = distance
                        recognized_name = row["name"]

                if min_distance < 0.68:
                    st.success(f"Bienvenue {recognized_name} (Distance : {min_distance:.4f})")

                    if recognized_name not in already_present:
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        new_entry = pd.DataFrame([[recognized_name, now, "Oui"]],
                                                 columns=["name", "datetime", "present"])
                        presence_df = pd.concat([presence_df, new_entry], ignore_index=True)
                        presence_df.to_excel(presence_file, index=False)
                        already_present.add(recognized_name)
                        st.write(f"Présence enregistrée pour {recognized_name} à {now}")
                    else:
                        st.info(f"{recognized_name} est déjà enregistré dans la feuille de présence.")
                else:
                    st.warning("Aucune correspondance trouvée.")

            except Exception as e:
                st.error(f"Erreur lors de la reconnaissance : {e}")

        if key == ord('q'):
            break

    cap.release()

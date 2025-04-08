import streamlit as st
import pandas as pd
import os
from deepface import DeepFace
import numpy as np
from PIL import Image
import io
from datetime import datetime

# Créer le dossier de sauvegarde si inexistant
save_path = "faces/"
os.makedirs(save_path, exist_ok=True)

csv_file = "faces.csv"
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
else:
    df = pd.DataFrame(columns=["name"] + [f"e{i}" for i in range(2622)])  # VGG-Face size

presence_file = "presence.xlsx"

# Charger les présences déjà enregistrées (si le fichier existe)
if os.path.exists(presence_file):
    presence_df = pd.read_excel(presence_file)
    already_present = set(presence_df["name"].tolist())
else:
    presence_df = pd.DataFrame(columns=["name", "datetime", "present"])
    already_present = set()

# Interface Streamlit
st.title("🎭 Système de Reconnaissance Faciale")

# Onglets : Ajouter un visage / Reconnaissance
tab1, tab2 = st.tabs(["➕ Ajouter un visage", "🔍 Reconnaissance"])

with tab1:
    st.header("Ajouter une personne")
    
    name = st.text_input("Entrez votre nom :")

    choice = st.radio(
        "Choisissez une méthode pour ajouter une image",
        ("Prendre une photo avec la webcam", "Charger une photo depuis le disque")
    )

    if choice == "Prendre une photo avec la webcam":
        image_file = st.camera_input("Prenez une photo")
        if image_file:
            image = Image.open(image_file)
            img_path = os.path.join(save_path, f"{name}.jpg")
            image.save(img_path)
            try:
                embedding = DeepFace.represent(img_path=img_path, model_name="VGG-Face", detector_backend="mtcnn")[0]["embedding"]
                
                new_data = pd.DataFrame([[name] + embedding], columns=df.columns)
                df = pd.concat([df, new_data], ignore_index=True)

                df.to_csv(csv_file, index=False)
                st.success(f"Encodage sauvegardé pour {name} dans {csv_file}")

            except Exception as e:
                st.error(f"Erreur lors de l'extraction des embeddings : {e}")

    elif choice == "Charger une photo depuis le disque":
        image_file = st.file_uploader("Téléchargez une photo", type=["jpg", "jpeg", "png"])
        if image_file:
            image = Image.open(image_file)
            img_path = os.path.join(save_path, f"{name}.jpg")
            image.save(img_path)
            try:
                embedding = DeepFace.represent(img_path=img_path, model_name="VGG-Face", detector_backend="mtcnn")[0]["embedding"]
                
                new_data = pd.DataFrame([[name] + embedding], columns=df.columns)
                df = pd.concat([df, new_data], ignore_index=True)

                df.to_csv(csv_file, index=False)
                st.success(f"Encodage sauvegardé pour {name} dans {csv_file}")

            except Exception as e:
                st.error(f"Erreur lors de l'extraction des embeddings : {e}")

with tab2:
    st.header("Reconnaissance Faciale")

    test_image = st.camera_input("Prenez une photo pour la reconnaissance")

    if test_image is not None:
        img = Image.open(test_image)
        img_path = "test.jpg"
        img.save(img_path)

        try:
            embedding = DeepFace.represent(
                img_path=img_path,
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
                st.success(f"✅ Bienvenue {recognized_name} (Distance : {min_distance:.4f})")

                if recognized_name not in already_present:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    new_entry = pd.DataFrame([[recognized_name, now, "Oui"]],
                                             columns=["Nom", "Heure", "Present"])
                    presence_df = pd.concat([presence_df, new_entry], ignore_index=True)
                    presence_df.to_excel(presence_file, index=False)
                    already_present.add(recognized_name)
                    st.write(f"Présence enregistrée pour {recognized_name} à {now}")
                else:
                    st.warning(f"{recognized_name} est déjà enregistré dans la feuille de présence.")
            else:
                st.warning("❌ Aucune correspondance trouvée.")

        except Exception as e:
            st.error(f"Erreur lors de la reconnaissance : {e}")

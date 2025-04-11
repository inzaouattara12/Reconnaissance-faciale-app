import streamlit as st
import pandas as pd
import os
from deepface import DeepFace
import numpy as np
from PIL import Image
from datetime import datetime
import hashlib
import io

# ---------- Authentification ----------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

AUTHORIZED_USERS = {
    "admin": hash_password("adminpass"),
    "prof": hash_password("monmotdepasse"),
}

def check_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = ""

st.title("🎭 Système de Reconnaissance Faciale")

check_login()

save_path = "faces/"
os.makedirs(save_path, exist_ok=True)

csv_file = "faces.csv"
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
else:
    df = pd.DataFrame(columns=["name"])

def get_presence_file():
    date_today = datetime.now().strftime("%d-%m-%Y")
    return f"presence_du_{date_today}.xlsx"

def delete_all_faces():
    for filename in os.listdir(save_path):
        file_path = os.path.join(save_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    if os.path.exists(csv_file):
        os.remove(csv_file)
    return "Tous les visages ont été supprimés."

def delete_presence():
    presence_file = get_presence_file()
    if os.path.exists(presence_file):
        os.remove(presence_file)
    return f"Liste de {presence_file} a été supprimée."

tab1, tab2 = st.tabs(["➕ Ajouter un visage", "🔍 Reconnaissance"])

with tab1:
    if not st.session_state.authenticated:
        st.warning("Veuillez vous connecter pour ajouter un visage.")
        login_form = st.form(key="login_form")
        username = login_form.text_input("Nom d'utilisateur", value=st.session_state.username)
        password = login_form.text_input("Mot de passe", type="password")
        submit_button = login_form.form_submit_button(label="Se connecter")

        if submit_button:
            if username and password:
                if username in AUTHORIZED_USERS and hash_password(password) == AUTHORIZED_USERS[username]:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success(f"Bienvenue {username} ! Vous êtes maintenant connecté.")
                    st.rerun()
                else:
                    st.error("Identifiants incorrects.")
            else:
                st.warning("Veuillez remplir tous les champs.")
    else:
        if st.button("Se déconnecter"):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.rerun()

        if st.button("Supprimer tous les visages enregistrés"):
            result = delete_all_faces()
            st.success(result)

        if st.button("Supprimer la liste de présence"):
            result = delete_presence()
            st.success(result)

        st.header("Ajouter une personne")
        name = st.text_input("Entrez votre nom :")
        choice = st.radio("Choisissez une méthode pour ajouter une image", 
                          ("Prendre une photo avec la webcam", "Charger une photo depuis le disque"))

        if choice == "Prendre une photo avec la webcam":
            image_file = st.camera_input("Prenez une photo")

            if image_file:
                if not name:
                    st.warning("❌ Veuillez entrer un nom avant de prendre la photo.")
                else:
                    image = Image.open(image_file).convert("RGB")
                    img_path = os.path.join(save_path, f"{name}.jpg")
                    image.save(img_path)
                    try:
                        embedding = DeepFace.represent(img_path=img_path, model_name="VGG-Face", detector_backend="mtcnn")[0]["embedding"]

                        if df.empty:
                            columns = ["name"] + [f"e{i}" for i in range(len(embedding))]
                            df = pd.DataFrame(columns=columns)

                        new_data = pd.DataFrame([[name] + embedding], columns=df.columns)
                        df = pd.concat([df, new_data], ignore_index=True)
                        df.to_csv(csv_file, index=False)
                        st.success(f"✅ Encodage sauvegardé pour {name}")
                    except Exception as e:
                        if 'Face could not be detected' in str(e):
                            st.error("❌ Aucun visage détecté. Veuillez vous assurer que la photo est bien centrée sur un visage clairement visible.")
                        else:
                            st.error(f"Erreur lors de la reconnaissance : {e}")

        elif choice == "Charger une photo depuis le disque":
            image_file = st.file_uploader("Téléchargez une photo", type=["jpg", "jpeg", "png"])
    
            if image_file:
                if not name:
                    st.warning("❌ Veuillez entrer un nom avant de télécharger la photo.")
                else:
                    image = Image.open(image_file).convert("RGB")
                    img_path = os.path.join(save_path, f"{name}.jpg")
                    image.save(img_path)
                    try:
                        embedding = DeepFace.represent(img_path=img_path, model_name="VGG-Face", detector_backend="mtcnn")[0]["embedding"]

                        if df.empty:
                            columns = ["name"] + [f"e{i}" for i in range(len(embedding))]
                            df = pd.DataFrame(columns=columns)

                        new_data = pd.DataFrame([[name] + embedding], columns=df.columns)
                        df = pd.concat([df, new_data], ignore_index=True)
                        df.to_csv(csv_file, index=False)
                        st.success(f"✅ Encodage sauvegardé pour {name}")
                    except Exception as e:
                        if 'Face could not be detected' in str(e):
                            st.error("❌ Aucun visage détecté. Veuillez vous assurer que la photo est bien centrée sur un visage clairement visible.")
                        else:
                            st.error(f"Erreur lors de la reconnaissance : {e}")


with tab2:
    st.header("Reconnaissance Faciale")
    presence_file = get_presence_file()

    if os.path.exists(presence_file):
        presence_df = pd.read_excel(presence_file)
    else:
        presence_df = pd.DataFrame(columns=["name", "Heure", "Present"])

    test_image = st.camera_input("Prenez une photo pour la reconnaissance")

    if test_image is not None:
        img = Image.open(test_image).convert("RGB")
        img_path = "test.jpg"
        img.save(img_path)

        try:
            embedding = DeepFace.represent(img_path=img_path, model_name="VGG-Face", detector_backend="mtcnn")[0]["embedding"]

            min_distance = float("inf")
            recognized_name = "Inconnu"

            for _, row in df.iterrows():
                stored_embedding = np.array(row[1:], dtype=np.float64)
                distance = np.linalg.norm(stored_embedding - embedding)

                if distance < min_distance:
                    min_distance = distance
                    recognized_name = row["name"]

            if min_distance < 0.68:
                st.success(f"✅ Bienvenue {recognized_name}")
                if recognized_name not in presence_df["name"].values:
                    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                    new_entry = pd.DataFrame([[recognized_name, now, "Oui"]], columns=["name", "Heure", "Present"])
                    presence_df = pd.concat([presence_df, new_entry], ignore_index=True)
                    presence_df.to_excel(presence_file, index=False)
                    st.write(f"Présence enregistrée pour {recognized_name} le {now}")
                else:
                    st.warning(f"{recognized_name} est déjà enregistré.")
            else:
                st.warning("❌ Aucune correspondance trouvée.")
        except Exception as e:
            if 'Face could not be detected' in str(e):
                st.error("❌ Aucun visage détecté. Veuillez vous assurer que la photo est bien centrée sur un visage clairement visible.")
            else:
                st.error(f"Erreur lors de la reconnaissance : {e}")

    if not presence_df.empty:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            presence_df.to_excel(writer, index=False)
        output.seek(0)
        st.download_button(
            label="📥 Télécharger la feuille de présence",
            data=output,
            file_name=presence_file,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

import streamlit as st
from transformers import pipeline
import pandas as pd
import requests

# ----------- Carga del modelo Zero-Shot una vez -----------
@st.cache_resource
def load_zero_shot_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


# ----------- Interfaz con pestañas (Ejercicio 1 y 2) -----------
tab1, tab2 = st.tabs(["🧠 Clasificador Zero-Shot", "🤖 Chatbot con Memoria"])


# ======================= EJERCICIO 1 =======================
with tab1:
    st.title("🧠 Clasificador de Tópicos Flexible (Zero-Shot)")
    st.write("""
    Esta aplicación clasifica un texto en las categorías que tú elijas, **sin necesidad de reentrenamiento**.
    Usa el modelo `facebook/bart-large-mnli` para realizar inferencia de lenguaje natural (NLI).
    """)

    text_input = st.text_area("✏️ Ingresa el texto que deseas analizar:", height=200)
    labels_input = st.text_input("🏷️ Ingresa las etiquetas (separadas por comas):", value="deportes, política, salud")

    if st.button("📊 Clasificar"):
        if text_input.strip() == "" or labels_input.strip() == "":
            st.warning("Por favor ingresa un texto y al menos una etiqueta.")
        else:
            with st.spinner("Analizando con el modelo..."):
                labels = [label.strip() for label in labels_input.split(",") if label.strip() != ""]
                classifier = load_zero_shot_model()
                result = classifier(text_input, candidate_labels=labels)

                scores = result['scores']
                labels = result['labels']

                df = pd.DataFrame({
                    "Etiqueta": labels,
                    "Puntaje": [round(score * 100, 2) for score in scores]
                })

                st.dataframe(df)
                st.bar_chart(df.set_index("Etiqueta"))


# ======================= EJERCICIO 2 =======================
with tab2:
    st.title("🤖 Chatbot Conversacional con Memoria (Groq + LLaMA3)")

    # Campo para ingresar la API key
    if "GROQ_API_KEY" not in st.session_state:
        st.session_state.GROQ_API_KEY = ""

    api_key_input = st.text_input(
        "🔐 Ingresa tu Groq API Key:",
        type="password",
        value=st.session_state.GROQ_API_KEY,
        help="Puedes obtenerla en https://console.groq.com/"
    )

    # Guardar la clave en la sesión si se ingresa
    if api_key_input:
        st.session_state.GROQ_API_KEY = api_key_input.strip()

    # Validar que la API key esté presente
    if not st.session_state.GROQ_API_KEY:
        st.warning("Por favor ingresa tu Groq API Key para usar el chatbot.")
    else:
        # Inicializar historial en session_state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role": "system", "content": "Eres un asistente útil que responde preguntas del usuario."}
            ]

        # Mostrar historial
        for msg in st.session_state.chat_history[1:]:  # omitimos el system prompt
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input del usuario
        user_input = st.chat_input("Escribe tu mensaje aquí...")

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Llamada a la API usando la clave ingresada
            with st.spinner("Pensando..."):
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {st.session_state.GROQ_API_KEY}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "llama3-8b-8192",
                    "messages": st.session_state.chat_history,
                    "temperature": 0.7
                }

                response = requests.post(url, headers=headers, json=payload)

                if response.status_code == 200:
                    reply = response.json()["choices"][0]["message"]["content"]
                else:
                    reply = f"❌ Error {response.status_code}: {response.text}"

            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

import streamlit as st
from transformers import pipeline
import pandas as pd
import requests
import os

# ----------- Carga del modelo Zero-Shot una vez -----------
@st.cache_resource
def load_zero_shot_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# ----------- Función para llamar a la API de Groq -----------
def call_groq_llm(messages):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.secrets['GROQ_API_KEY']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-8b-8192",
        "messages": messages,
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"❌ Error {response.status_code}: {response.text}"


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
        # Agregar el mensaje del usuario
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Mostrar el mensaje del usuario
        with st.chat_message("user"):
            st.markdown(user_input)

        # Llamar a Groq
        with st.spinner("Pensando..."):
            assistant_reply = call_groq_llm(st.session_state.chat_history)

        # Agregar respuesta del modelo
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})

        # Mostrar respuesta
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)

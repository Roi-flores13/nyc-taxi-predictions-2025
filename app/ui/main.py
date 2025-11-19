import streamlit as st
import requests
import json

st.write("""
# Application to predict the time for the NYC taxi trips
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    PU = st.sidebar.text_input("PU Location ID", "80")
    DO = st.sidebar.text_input("DO Location ID", "60")
    trip_distance = st.sidebar.number_input("Trip Distance", value=10.0, min_value=0.11, max_value=100.0)

    input_dict = {
        'PULocationID': PU,
        'DOLocationID': DO,
        'trip_distance': trip_distance,
    }

    return input_dict

input_dict = user_input_features()

if st.button('Predict'):
    # 1. FIX: La URL ahora coincide con tu API (v9)
    url = "http://127.0.0.1:8000/api/v9/predict" 
    
    # 2. FIX: Usa el parámetro 'json' para enviar los datos correctamente
    response = requests.post(
        url=url,
        json=input_dict 
    )

    # --- Mejora Opcional pero Recomendada ---
    # Comprueba si la petición fue exitosa (código 200)
    if response.status_code == 200:
        prediction_data = response.json()
        st.write(f"El tiempo estimado del viaje es: {prediction_data['prediction']} minutos")
    else:
        # Muestra el error de la API en la app
        st.error(f"Error al llamar a la API: {response.status_code}")
        st.json(response.json()) # Muestra el JSON del error (ej: {'detail': ...})
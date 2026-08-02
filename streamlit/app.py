import streamlit as st

st.set_page_config(page_title="Buscador de direcciones", layout="wide")
st.title("Inicio")

st.write("Elegí una página desde la barra lateral o desde estos accesos rápidos:")

# Opcional: accesos directos (si tu versión de Streamlit soporta st.page_link)
try:
    st.page_link("pages/01_unstructured.py", label="Modelo desestructurado", icon="🔎")
    st.page_link("pages/02_structured.py", label="Modelo estructurado", icon="📚")
except Exception:
    st.info("Usá el selector de páginas en la barra lateral.")

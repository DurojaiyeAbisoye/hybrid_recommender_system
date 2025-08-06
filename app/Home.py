# app/Home.py
import streamlit as st

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


st.set_page_config(page_title="Recommender Entry", layout="centered")
st.title("🛍️ Welcome to the Recommender System")

st.markdown("### Are you an existing user or a new user?")
option = st.radio("Select user type:", ["Existing User", "Cold-Start User"])

# Save to session_state so 1_Webstore.py can access the selection
st.session_state.user_type = option

# Navigation link to the webstore recommendation page
st.page_link("pages/1_Webstore.py", label="Go to Recommendations →", icon="➡️")



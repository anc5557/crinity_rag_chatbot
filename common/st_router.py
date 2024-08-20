import streamlit as st
from common.st_initializer import initialize_router


@st.cache_resource
def get_router():
    return initialize_router()

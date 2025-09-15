import streamlit as st

def show_msg(msg: str, type="info"):
    """Convenience wrapper for Streamlit messages."""
    if type == "info":
        st.info(msg)
    elif type == "success":
        st.success(msg)
    elif type == "error":
        st.error(msg)
    elif type == "warning":
        st.warning(msg)

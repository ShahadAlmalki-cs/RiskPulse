import inspect
import streamlit as st
import streamlit.delta_generator as dg

for name in ["metric", "columns", "text_input", "selectbox", "badge", "button", "container"]:
    print(name, inspect.signature(getattr(st, name)))
print("sidebar", inspect.signature(dg.DeltaGenerator.__getattribute__) if False else None)
print([n for n in dir(st.sidebar) if not n.startswith("_")][:10])
print(inspect.signature(st.sidebar.__getattr__("__call__")) if False else "")

import streamlit as st
from fastai.learner import load_learner


filename = "\model.pkl"
model = load_learner(filename)


def main():
    st.set_page_config(page_title="Zadanie 5 - Piotr Trzos")
    st.title("Zadanie 5 - Piotr Trzos")
    st.subheader("Cat/Dog Image classifier")
    uploaded_file = st.file_uploader("Choose an image of cat/dog", accept_multiple_files=False, type=["jpg", "png", "JPEG"])
    if uploaded_file:
        st.image(uploaded_file, use_column_width=True)
        prediction = model.predict(uploaded_file.read())[0]
        st.write(prediction)


if __name__ == "__main__":
    main()

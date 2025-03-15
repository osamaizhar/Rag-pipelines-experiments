import streamlit as st
import import_ipynb
from groq_test import process_user_query

# Streamlit interface
def run_streamlit_app():
    st.title("AI Coach - Query Processor")

    # Input box for the user's query
    user_query = st.text_input("Enter your query:")

    # Button to submit the query
    if st.button("Submit"):
        if user_query:
            # Call the function from the notebook
            response = process_user_query(user_query)
            st.write("AI Coach's Response:")
            st.write(response)
        else:
            st.write("Please enter a query.")

# Running the Streamlit app
if __name__ == "__main__":
    run_streamlit_app()

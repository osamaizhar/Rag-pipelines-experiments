# from query_llm import process_user_query

# print(process_user_query())


from importnb import Notebook

with Notebook():
    from groq_test import process_user_query


print(process_user_query())
# import os
# # from langchain_community.llms import Groq

# # class GroqLLMWrapper:
# #     def __init__(self, model: str = "llama3-70b-8192"):
# #         self.llm = Groq(
# #             api_key=os.getenv("GROQ_API_KEY"),
# #             model_name=model,
# #             temperature=0.7,
# #             max_tokens=2048
# #         )

# #     def __call__(self, prompt: str) -> str:
# #         return self.llm(prompt)

# from dotenv import load_dotenv
# load_dotenv()
# import os

# from langchain_groq import ChatGroq


# #os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
# os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")


# # llm=ChatGroq(model="qwen-2.5-32b")
# #llm = ChatOpenAI(model="gpt-4o")
# # result=llm.invoke("Hello")
# # result
# from dotenv import load_dotenv
# import os
# from langchain_groq import ChatGroq

# load_dotenv()
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# class GroqLLMWrapper:
#     def __init__(self, model: str = "qwen-2.5-32b"):
#         self.llm = ChatGroq(model=model)

#     def generate(self, prompt: str) -> str:
#         result = self.llm.invoke(prompt)
#         # If result is a LangChain message object, extract content
#         if hasattr(result, "content"):
#             return result.content
#         return str(result)

# # Example usage:
# # llm = GroqLLMWrapper()
# # proposal = llm.generate("Write a grant proposal for AI in healthcare.")
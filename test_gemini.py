import google.generativeai as genai

genai.configure(api_key="AIzaSyBR_t45_GWHlBivvmhM0iOUk7Zl0JeplKE")

model = genai.GenerativeModel("gemini-1.5-flash-latest")
response = model.generate_content("Say hello")

print(response.text)

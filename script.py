from google import genai

with open("knowledge.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

chunks = [p.strip() for p in raw_text.split("\n\n") if p.strip()]

def create_prompt(query, context):
    prompt = f"""
        Answer the user's question using ONLY the provided context below. If the answer cannot be found in the context, reply 'I don't know based on the provided context'

        Context:
        {context}

        Question:
        {query}
    """
    client = genai.Client(api_key="removed")

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    return response.text;

user_question = "What is my name?"
print(create_prompt(user_question, chunks))


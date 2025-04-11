import ollama

model = ollama.Model('llama2')

def ask_fashion_bot(prompt):
    response = model.generate(prompt)
    return response

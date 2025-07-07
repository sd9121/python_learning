# Simulated LLM response function (you can replace this with a real API call)
def llm_response(prompt):
    responses = {
        "what is a list in python": "A list in Python is an ordered collection of items, defined using square brackets, e.g., [1, 2, 3].",
        "how to write a for loop": "Use 'for item in list:' to loop over items. Example: for x in [1, 2, 3]: print(x)",
        "how to define a function": "Use 'def' to define a function. Example: def greet(): print('Hello')"
    }
    return responses.get(prompt.lower(), "Sorry, I don't know the answer to that yet.")

# Learning loop
questions = [
    "What is a list in Python",
    "How to write a for loop",
    "How to define a function"
]

print("🧠 AI Learning Assistant\n")

for question in questions:
    print(f"👨‍🎓 You: {question}")
    answer = llm_response(question)
    print(f"🤖 AI: {answer}\n")

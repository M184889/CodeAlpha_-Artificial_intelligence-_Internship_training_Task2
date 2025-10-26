import tkinter as tk
from tkinter import scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample FAQs (You can edit or add more)
faqs = {
    "What is your return policy?": "Our return policy allows returns within 30 days of purchase.",
    "How can I track my order?": "You can track your order using the tracking ID sent to your email.",
    "Do you offer international shipping?": "Yes, we offer international shipping to selected countries.",
    "What payment methods are accepted?": "We accept credit cards, PayPal, and bank transfers.",
    "How can I contact customer support?": "You can contact our support team via email or live chat."
}

# Extract questions and answers
questions = list(faqs.keys())
answers = list(faqs.values())

# Create TF-IDF model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# Chatbot logic
def get_response(user_input):
    user_input_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_input_vector, X)
    best_match_index = similarities.argmax()
    return answers[best_match_index]

# GUI Setup
root = tk.Tk()
root.title("FAQ Chatbot")
root.geometry("500x500")
root.config(bg="#f2f2f2")

chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20, font=("Arial", 10))
chat_area.pack(padx=10, pady=10)
chat_area.config(state=tk.DISABLED)

entry_field = tk.Entry(root, width=60, font=("Arial", 10))
entry_field.pack(padx=10, pady=10)

def send_message():
    user_input = entry_field.get()
    if user_input.strip() == "":
        return
    chat_area.config(state=tk.NORMAL)
    chat_area.insert(tk.END, "You: " + user_input + "\n")
    if user_input.lower() == "exit":
        chat_area.insert(tk.END, "Chatbot: Goodbye! 👋\n")
        root.after(1000, root.destroy)
    else:
        response = get_response(user_input)
        chat_area.insert(tk.END, "Chatbot: " + response + "\n\n")
    chat_area.config(state=tk.DISABLED)
    chat_area.see(tk.END)
    entry_field.delete(0, tk.END)

send_button = tk.Button(root, text="Send", command=send_message, font=("Arial", 10, "bold"), bg="#4CAF50", fg="white")
send_button.pack(pady=5)

chat_area.config(state=tk.NORMAL)
chat_area.insert(tk.END, "Chatbot: Hello! Ask me anything about our services. (Type 'exit' to quit)\n\n")
chat_area.config(state=tk.DISABLED)

root.mainloop()

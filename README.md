# 💬 RAG Chatbot  

A **Retrieval-Augmented Generation (RAG) Chatbot** built with `LangChain`, `FAISS`, and `Streamlit`. This chatbot can process various types of inputs, store them in a FAISS vector store, and generate responses using **Meta Llama 3 8B Instruct**.

---

## Features  
- 📂 Supports multiple input types: **Text, PDFs, DOCX, TXT, and URLs**  
- ⚡ Uses **FAISS** for efficient similarity search  
- 🤖 Utilizes **Meta Llama 3 8B** for response generation  
- 🔍 Processes documents with **HuggingFace Embeddings**  
- 🌐 Streamlit-based interactive UI  

---

## Installation  

### **1️⃣ Clone the Repository**  
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/SHIGINVP/RAG_chatbot)
cd YOUR_REPOSITORY
```

### **2️⃣ Create a Virtual Environment (Optional but Recommended)** 

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### **3️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```


### **Usage**

## Run the Chatbot

```bash
streamlit run app.py
```


## Using the Chatbot

1. Select an input type (Text, PDF, DOCX, TXT, or URL).

2. Provide the input data (upload a file or paste a URL/text).

3. Click the "Process Data" button to create the FAISS vector store.

4. Start asking questions in the chat!


## Dependencies

1. `Python 3.8+`

2. `streamlit`

3. `langchain`

4. `faiss-cpu`

5. `huggingface-hub`

6. `PyPDF2`

7. `docx`

8. `numpy`



## License

MIT License


## 📌 Future Enhancements

- ✅ Support for more document types

- ✅ UI improvements

-✅ Faster response times


This README explains how to install, run, and use your RAG Chatbot. Let me know if you need modifications! 🚀


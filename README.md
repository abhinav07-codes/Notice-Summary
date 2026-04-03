# Notice-Summary
College Notices made easy!
📄 Academic Notice Summarizer

🚀 Automated Document Processing Pipeline for Academic Notices

An AI-powered system that processes academic notices (PDFs & images) and generates structured, concise summaries — saving students from digging through lengthy documents.

🧠 Overview

Academic institutions release notices in multiple formats — often dense and hard to read. This project builds a multi-stage pipeline that:

Classifies documents (PDF/Image)
Extracts text (OCR + parsing)
Identifies structured information (dates, title, etc.)
Generates AI-powered summaries

All of this is wrapped inside an interactive Streamlit web app.

🎯 Features

✅ Supports:

Digital PDFs
Scanned PDFs
Image files

✅ Intelligent pipeline:

Document classification
OCR-based extraction
Rule-based structure parsing
LLM-based summarization

✅ Smart UI:

Page selector for multi-page documents
Structured summary output
Downloadable JSON
🏗️ System Architecture
Upload → Classify → Extract → Parse → Summarize → Display
Modules:
Document Classifier
Detects: digital_pdf, scanned_pdf, image
Text Extractor
PyMuPDF + pdfplumber (PDF)
EasyOCR (scanned/image)
Structure Parser
Extracts:
Title
Issuing body
Dates
Notice number
LangChain Summarizer
LLM generates structured summary
Uses Pydantic schema validation
Streamlit App
User interface + visualization
🛠️ Tech Stack
Technology	Role
Python 3.11	Core language
PyMuPDF	PDF text extraction
pdfplumber	Table extraction
EasyOCR	OCR for scanned docs
LangChain	LLM orchestration
Groq API	Fast LLM inference
Pydantic	Structured output validation
Streamlit	Web UI
📦 Installation
git clone https://github.com/your-username/notice-summarizer.git
cd notice-summarizer

pip install -r requirements.txt
🔑 Environment Variables

Create a .env file:

GROQ_API_KEY=your_api_key_here
▶️ Run the App
streamlit run app.py
📊 Example Output

Input: Quiz-II Time Table (Scanned PDF)

Output:

📌 One-line summary
📅 Important dates
👥 Target audience
⚠️ Urgency level
📝 Key points
📚 Subject-wise schedule
⚡ Key Highlights
Handles multi-page documents intelligently
Prevents content mixing across pages
Uses hybrid approach (rules + AI) for accuracy
Fast inference (~2 seconds with Groq)
🧪 Performance
✅ High OCR accuracy (EasyOCR)
✅ Correct date normalization (ISO format)
✅ Accurate structured summaries via LLM
⚡ Cached results for faster re-use
⚠️ Challenges Solved
Windows compatibility issues (OCR tools)
Multi-page document confusion
Streamlit rerun state management
LLM schema validation errors
🔮 Future Improvements
🌐 Multilingual support (Hindi + regional)
📲 WhatsApp/Email notifications
☁️ Cloud deployment
🧠 Fine-tuned local LLM
📊 Better table extraction for scanned docs
🤝 Contribution

Feel free to fork, improve, and submit PRs 🚀

📜 License

MIT License

👨‍💻 Author

Abhinav Saha
B.Tech CSE — Sikkim Manipal Institute of Technology

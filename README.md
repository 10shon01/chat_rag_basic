# Document-Based Q&A System

This project is a simple yet effective Q&A system that leverages Chainlit, LangChain, and OpenAI GPT-4o. It enables users to upload documents and ask questions, retrieving precise answers based on the document's content.

---

## Key Features

### Document Upload
- Users can upload PDF, Word, or Excel files for processing.
- Supports diverse file formats for versatility in document types.

### Text Extraction & Embedding
- Extracts text from uploaded files.
- Processes extracted text into vector embeddings using OpenAI's embeddings model.

### Semantic Search
- Employs FAISS to perform similarity searches, retrieving the most relevant sections of the documents.

### Answer Generation
- Utilizes OpenAI's GPT-4o to generate contextually accurate answers based on the retrieved document sections.

### User Feedback Loop
- Allows users to provide feedback (e.g., 👍 or 👎) on the quality of responses, supporting iterative improvement.

### Security Measures

---

## How It Works

1. Enter 'ファイルアップロード'.
2. Users upload a document (PDF, Word, or Excel).
3. The system extracts and preprocesses the document content.
4. The content is embedded into a vector store using FAISS.
5. When users ask questions, the system:
   - Performs a semantic search to find the most relevant sections.
   - Sends the retrieved sections along with the user query to OpenAI's GPT-4o.
6. The generated answer is displayed along with the sources.

# ScaffoldAI Enhanced UI v0.1.0

## For Professors - Quick Start Guide

### 1. Download and Setup
1. **Download the EXE** from this release
2. **Create a `.env` file** in the same folder as the EXE with:
   ```
   HUGGINGFACE_TOKEN=your_token_here
   SC_LLM_KEY=tinyllama-onnx
   ```
3. **Run the EXE**: Double-click or run from command line
4. **Open browser** to http://localhost:5002

### 2. Features
- **Syllabus Upload**: Upload PDF syllabi for personalized sustainability suggestions
- **Chat Interface**: Ask questions about integrating sustainability into engineering courses
- **Research-Backed Answers**: All responses include citations from sustainability education literature
- **Session Memory**: Conversations are saved and can be continued across sessions
- **Source Citations**: View the research papers and documents that inform each answer

### 3. How to Use
1. **Upload Syllabus**: Use the sidebar to upload your course syllabus PDF
2. **Ask Questions**: Type questions like:
   - "How can I integrate sustainability into my fluid mechanics course?"
   - "What are some hands-on sustainability projects for thermodynamics?"
   - "Show me examples of sustainable design in materials engineering"
3. **Review Sources**: Click on source citations to see the research behind answers
4. **Continue Conversations**: The system remembers your syllabus and previous questions

### 4. System Requirements
- Windows 10/11
- Internet connection (for initial model downloads)
- ~2GB RAM available
- Port 5002 must be available

### 5. Troubleshooting
- **EXE won't start**: Ensure `.env` file is in the same folder as the EXE
- **Windows Defender blocks**: Add the EXE folder to Windows Defender exclusions
- **Port in use**: Change the port in the `.env` file: `SC_PORT=5003`
- **Model download fails**: Check your internet connection and Hugging Face token

### 6. Example Questions to Try
- "What sustainability concepts can I add to my engineering design course?"
- "How do I assess student learning of sustainability principles?"
- "What are some industry examples of sustainable engineering practices?"
- "How can I make sustainability relevant to mechanical engineering students?"

### 7. Getting Help
- Check the browser console (F12) for error messages
- Ensure your Hugging Face token is valid
- The system works best with specific course-related questions

---

**Note**: This is a research prototype. For production use, consider hosting on a server or using the web version.

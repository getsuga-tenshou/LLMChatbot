# LLMChatbot
LLM-Powered Chatbot with Entity Linking and Fact-Checking

## Overview
This project is a command-line-based chatbot that leverages a Large Language Model (LLM) to answer user queries, using the **Llama 3B** model. It integrates natural language processing (NLP) with **spaCy** and utilizes the **Wikipedia API** for fact-checking and entity linking. The chatbot generates responses, extracts relevant information, and validates the correctness of its answers against trusted external knowledge bases, enhancing both the accuracy and reliability of the chatbot's output.

## Features
- **LLM Integration**: Uses the **Llama 3B** model for generating coherent and contextually relevant responses.
- **Entity Linking**: Identifies and links key entities in both the query and response to relevant Wikipedia pages.
- **Fact-Checking**: Verifies the correctness of the generated responses by cross-referencing them with external sources like Wikipedia.
- **Disambiguation Handling**: Utilizes NLP techniques to resolve ambiguous entities by analyzing context and selecting the most relevant Wikipedia entry.

## Technology Stack
- **Llama 3B (LLM)**: A compact large language model from **Hugging Face** used for text generation.
- **spaCy**: An NLP library for entity extraction, similarity scoring, and disambiguation.
- **Wikipedia API**: For fetching entity information and verifying facts from Wikipedia.
- **Transformers (Hugging Face)**: Used for loading and generating text from the Llama 3B model.
- **Torch**: For handling GPU and tensor computations.

## Example Interaction
### User Query:
`Is Managua the capital of Nicaragua?`

### Generated Response:
```
"Most people think Managua is the capital of Nicaragua. However, Managua is not the capital of Nicaragua. The capital of Nicaragua is Managua."
```

### Extracted Answer:
`"Managua"`

### Correctness:
`"Correct"`

### Entity Links:
- Managua: [Wikipedia](https://en.wikipedia.org/wiki/Managua)
- Nicaragua: [Wikipedia](https://en.wikipedia.org/wiki/Nicaragua)

## Setup and Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/llm-chatbot.git
   cd llm-chatbot
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the required models:**
   - **spaCy NLP model**:  
     ```bash
     python -m spacy download en_core_web_md
     ```
   - **Llama 3B model** from Hugging Face:
     ```bash
     transformers-cli download openlm-research/open_llama_3b
     ```

4. **Run the chatbot:**
   ```bash
   python chatbot.py
   ```

## Future Improvements
- **Enhanced Disambiguation**: Further improve disambiguation by leveraging more advanced context analysis methods.
- **Extended Knowledge Base**: Integrate additional knowledge bases beyond Wikipedia to cover a broader range of topics.
- **Improved LLM Model**: Upgrade to a larger or more fine-tuned language model such as GPT-4 or Llama 2 for enhanced response accuracy and coherence in complex queries.
```

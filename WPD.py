import spacy
from transformers import AutoModelForCausalLM, AutoTokenizer
from SPARQLWrapper import SPARQLWrapper, JSON
import requests
import torch
import re

# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForCausalLM.from_pretrained('openlm-research/open_llama_3b').to(device)

# Load spaCy for NLP tasks
nlp = spacy.load('en_core_web_md')

# Functions for entity extraction and disambiguation
def extract_and_link_entities(question, raw_answer):
    """
    Extracts entities from the question and raw answer, and links them to Wikipedia.
    """
    question_doc = nlp(question)
    answer_doc = nlp(raw_answer)
    linked_entities = {}

    for doc in [question_doc, answer_doc]:
        for entity in doc.ents:
            wikipedia_link, page_title = search_wikipedia(entity.text, doc.text)
            linked_entities[entity.text] = (entity.text, entity.label_, wikipedia_link) 

    return linked_entities


def search_wikipedia(entity, context):
    session = requests.Session()
    URL = "https://en.wikipedia.org/w/api.php"
    PARAMS = {
        'action': "query",
        'list': "search",
        'srsearch': entity,
        'format': "json",
        'srlimit': 1
    }

    response = session.get(url=URL, params=PARAMS)
    data = response.json()

    search_results = data['query']['search']
    if search_results:
        page_title = search_results[0]['title']
        page_id = search_results[0]['pageid']

        # Check if the page is a disambiguation page
        disambiguation_check_params = {
            'action': "query",
            'prop': "pageprops",
            'pageids': page_id,
            'format': "json"
        }

        disambiguation_check_response = session.get(url=URL, params=disambiguation_check_params)
        disambiguation_check_data = disambiguation_check_response.json()
        page_props = disambiguation_check_data['query']['pages'][str(page_id)]['pageprops']

        if "disambiguation" in page_props:
            # Handle disambiguation
            return handle_disambiguation(page_title, entity, context)

        return f"https://en.wikipedia.org/?curid={page_id}", page_title
    else:
        return "No Wikipedia page found", None, None



def fetch_summary(page_title):
    """Fetch the summary of a Wikipedia page."""
    summary_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + page_title.replace(' ', '_')
    response = requests.get(summary_url)
    if response.status_code == 200:
        data = response.json()
        return data.get('extract', '')
    return ''

def compute_relevance_score(summary, context, nlp_model):
    """Compute the relevance score based on NLP analysis."""
    doc1 = nlp_model(summary)
    doc2 = nlp_model(context)
    return doc1.similarity(doc2)

def handle_disambiguation(disambiguation_title, entity, context):
    """Handle disambiguation pages using NLP for context analysis."""
    session = requests.Session()
    disambiguation_url = f"https://en.wikipedia.org/w/api.php"
    disambiguation_params = {
        'action': "parse",
        'page': disambiguation_title,
        'format': "json",
        'prop': 'links'
    }
    response = session.get(url=disambiguation_url, params=disambiguation_params)
    disambiguation_data = response.json()

    best_match = None
    highest_score = -1

    if 'parse' in disambiguation_data:
        links = disambiguation_data['parse']['links']
        
        for link in links:
            if 'exists' in link and 'ns' in link and link['ns'] == 0:
                link_title = link['*']
                summary = fetch_summary(link_title)
                score = compute_relevance_score(summary, context, nlp)

                if score > highest_score:
                    best_match = link_title
                    highest_score = score

        if best_match:
            return f"https://en.wikipedia.org/wiki/{best_match.replace(' ', '_')}"

    return "No suitable page found on disambiguation page"


# Initialize LLaMA model
model_name = 'openlm-research/open_llama_3b'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_text_with_llama(prompt, max_length=50, temperature=0.7, num_return_sequences=1):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=num_return_sequences,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)




def extract_answer(model_output, question):
    """
    Extracts the answer from the LLaMA model's output based on the question.
    """
    # Process the model output with spaCy
    doc = nlp(model_output)
    question_doc = nlp(question)

    # Determine if the question is yes/no type
    if question.lower().startswith(('is', 'are', 'was', 'were', 'do', 'does', 'did', 'can', 'could', 'will', 'would', 'should', 'has', 'have', 'had')):
        # Implement logic to extract a yes/no answer
        for sentence in doc.sents:
            if sentence.text.strip().lower() in ['yes', 'no']:
                return sentence.text.strip()
        # If no clear yes/no found, consider the context to determine the answer
        # Placeholder for context determination logic
        return "Yes/No cannot be determined from the context"

    # Find the sentence that most likely contains the answer
    most_relevant_sentence = max(doc.sents, key=lambda s: s.similarity(question_doc))

    # Extract entities and noun chunks from the most relevant sentence
    entities = [ent.text for ent in most_relevant_sentence.ents if ent.root.dep_ in ('attr', 'dobj', 'pobj') and ent.root.head.lower_ in question.lower()]
    noun_chunks = [chunk.text for chunk in most_relevant_sentence.noun_chunks if chunk.root.dep_ in ('attr', 'dobj', 'pobj') and chunk.root.head.lower_ in question.lower()]

    # Combine entities and noun chunks, preferring entities
    all_candidates = entities + [chunk for chunk in noun_chunks if chunk not in entities]

    # Return the first candidate that seems to be an answer
    for candidate in all_candidates:
        if candidate.lower() != question_doc[0].text.lower():  # Avoid returning the question's subject
            return candidate

    # Fallback to the root of the sentence if no entity or noun chunk is found
    return most_relevant_sentence.root.text




def verify_answer_correctness(extracted_answer, question, nlp_model):
    """
    Verifies the correctness of the extracted answer by querying Wikipedia. 
    """
    # Extract the main subject from the question for Wikipedia search
    main_subject = extract_main_subject(question)

    # Get the Wikipedia page for the main subject
    wikipedia_url, wikipedia_data = search_wikipedia(main_subject, question)

    # If Wikipedia data is not found, return 'indeterminate'
    if not wikipedia_data:
        return "indeterminate"

    # Check if the extracted answer is in the summary of the Wikipedia page
    summary = fetch_summary(main_subject)
    if summary:
        relevance_score = compute_relevance_score(summary, extracted_answer, nlp_model)
        
        # Determine correctness based on relevance score
        # You can adjust the threshold as needed
        relevance_threshold = 0.5
        if relevance_score > relevance_threshold:
            return "correct"

    # Fallback to string matching if relevance score is low or not computed
    if extracted_answer.lower() in wikipedia_data.lower():
        return "correct"

    return "incorrect"

def extract_main_subject(question):
    """
    Extracts the main subject from the question.
    """
    doc = nlp(question)
    
    # Priority 1: Look for named entities first
    if doc.ents:
        return doc.ents[0].text

    # Priority 2: Look for noun chunks
    for noun_chunk in doc.noun_chunks:
        return noun_chunk.text

    # Priority 3: Look for individual nouns
    for token in doc:
        if token.pos_ == "NOUN" or token.pos_ == "PROPN":
            return token.text

    return None




# Function to format the output
def format_output(question_id, raw_answer, extracted_answer, correctness, entities_info):
    """
    Formats the output according to the requirements specified in the assignment.
    """
    formatted_output = f"{question_id}\tR\"{raw_answer}\"\n"
    formatted_output += f"{question_id}\tA\"{extracted_answer}\"\n"
    formatted_output += f"{question_id}\tC\"{correctness}\"\n"
    for entity, (text, label, link) in entities_info.items():  # Note the change here
        formatted_output += f"{question_id}\tE\"{text}\"\t\"{link}\"\n"
    return formatted_output

def process_input_output(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        current_question_id = None
        current_question_text = []

        for line in infile:
            print(f"Reading line: {line.strip()}")  # Debugging line
            if re.match(r"question-0\d{2}\t", line):
                if current_question_id:
                    # Process previous question
                    full_question_text = ' '.join(current_question_text)
                    print(f"Processing question: {current_question_id}")  # Debugging line
                    raw_answer = generate_text_with_llama(full_question_text)
                    print(f"LLaMA output: {raw_answer}")  # Debugging line
                    entities_info = extract_and_link_entities(full_question_text, raw_answer)
                    extracted_answer = extract_answer(raw_answer, full_question_text)
                    correctness = verify_answer_correctness(extracted_answer, full_question_text, nlp)
                    formatted_data = format_output(current_question_id, raw_answer, extracted_answer, correctness, entities_info)
                    outfile.write(formatted_data)

                current_question_id, question_text = line.strip().split('\t')
                current_question_text = [question_text]
            else:
                current_question_text.append(line.strip())

        # Process the last question
        if current_question_id:
            full_question_text = ' '.join(current_question_text)
            print(f"Processing last question: {current_question_id}")  # Debugging line
            raw_answer = generate_text_with_llama(full_question_text)
            entities_info = extract_and_link_entities(full_question_text, raw_answer)
            extracted_answer = extract_answer(raw_answer, full_question_text)
            correctness = verify_answer_correctness(extracted_answer, full_question_text, nlp)
            formatted_data = format_output(current_question_id, raw_answer, extracted_answer, correctness, entities_info)
            outfile.write(formatted_data)

def main():
    input_file = 'input.txt'
    output_file = 'output.txt'
    process_input_output(input_file, output_file)

if __name__ == "__main__":
    main()
import os
from openai import OpenAI
import gradio as gr
import uuid
import chromadb
from pprint import pprint
import spaces
import requests
import random
import json

#----------------------------------------------
# Setup
#----------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise Exception("API Key is missing") 
client = OpenAI()

#----------------------------------------------
# Document
#----------------------------------------------

document_overview = """
Yuan-Hung got his phD in Biochemistry from University of Illinois by study the molecular bining kintetics of the adhesion protein 
cadherin. After that he did his post-doc at Salk Institute by studying the importantce of mechanical force for breaking left-right 
symmetry in earyl embryonic development. His works contributed to several high profile-journal articles and presentations in cell
biology and developmental biology conference. But also at that time he started analzying NGS data like RNAseq and become a 
bioinformatics scientist. He also started programming in python.

After Salk, he start worked at the biotech industry as a data scientist/bioinformatics scientit. He is skilled in building machine 
learning and statistical models for biological and biomedical related data, especailly the oncology multi-omic data. 

Yuan-Hung like running, reading, cooking and food. He already ran 4 half-marathons and he also ran regulary everyweek. His favorite 
writers include Haruki Murakami, Roberto Bolano, Jorge Luis Borges, and Karl Ove Kaansgaard. He also enjoys red wine and Scottish 
Whiskey.  

Additional info:
-- In 2001, Yuan-Hung start to listen to a lot of classical music",
-- Yuan-Hung enjoy cooking and love to watch cooking video on Youtube. He also try new recipes regularly",
-- Yuan-Hung loves pineapple since childhood. He like to eat pineapple during the summer time",
-- Yuan-Hung loves pizza. But his wife is not a big fan. Pizza and red-wine make Yuan-Hung relax."
"""

document_education = """
Doctor of Philosophy (Ph.D.), Biochemistry, University of Illinois | Urbana, IL | GPA: 3.8/4.0 | Dissertation: Cadherin Recognition and Specificity
Master of Science (M.S.), Microbiology & Immunology, National Yang-Ming University | Taipei, Taiwan | 1999 | GPA: 4.0/4.0 | Thesis: Characterization of Hepatitis B Virus Negative Regulator NREBP
Bachelor of Science (B.S.), Life Science, Minor: Chemistry, National Tsing Hua University | Hsinchu, Taiwan | GPA: 3.8/4.0
"""

document_professional_experience= """
Principal Scientist, Bioinformatics  |  Boundless Bio |  La Jolla, CA
July 2025 – July 2026
Advance computational strategies supporting precision oncology research by integrating bioinformatics, machine learning, and cancer genomics to identify biomarkers, strengthen translational research, and inform therapeutic development. Partner with multidisciplinary scientists to transform complex drug-response and genomic datasets into actionable biological insights while expanding scalable analytical capabilities supporting oncology drug discovery programs.
●	Uncovered genomic features associated with therapeutic sensitivity for precision oncology research by analyzing preclinical drug-screening datasets, encompassing 800+ cancer cell lines across multiple tumor types.
●	Identified chromosomal regions associated with breast cancer sample response to a novel drug developed by Boundless Bio, using literature-guided feature selection and unsupervised machine learning, supporting patient stratification for the therapy.
●	Developed a scalable Large-Scale State Transition (LST) workflow by integrating NGS pipelines with HPC infrastructure, generating genomic instability scores aligned with internal and public benchmark datasets.
●	Fine-tuned Google's TxGemma large-scale genomic foundation model in Azure using LoRA and supervised fine-tuning (SFT) via the Hugging Face ecosystem, grounding a pretrained model in empirical benchmark data to improve genetically matched cell-line selection for investigational PROTAC therapeutics.
●	Enhanced bioinformatics workflows for whole-genome, whole-exome, and RNA sequencing, enabling ecDNA detection, allele-specific copy number profiling, and homologous recombination deficiency (HRD) assessment.
●	Analyzed disease-free survival outcomes in the MSK MetTropisim clinical trial cohort (with focus on breast, NSCLC, and ovarian cancer), using Kaplan-Meier estimation and log-rank testing (Python lifelines) to characterize differential survival between TP53 R273 and Y220 mutation subtypes.
●	Organized and analyzed whole-genome sequencing (WGS) data from patients enrolled in clinical trials, performing ecDNA status analysis to inform precision medicine strategy for the Precision Medicine Director.



Multiple Positions | Certis Oncology Solutions | San Diego, CA
June 2020 – July 2025

Principal Data Scientist   March 2025 – July 2025
Provided scientific leadership for advanced machine learning initiatives supporting precision oncology and targeted therapeutic development. Collaborated with cross-functional research teams to advance AI-driven approaches for drug discovery while translating computational analyses into actionable scientific insights that informed research strategy and therapeutic evaluation.
●	Established a structured data engineering workflow for antibody-drug conjugate (ADC) research by curating and integrating public datasets to accelerate machine learning development.
●	Presented machine learning-guided drug combination research at the AACR Special Conference on Artificial Intelligence and Machine Learning, advancing precision oncology and translational drug discovery.
●	Evaluated emerging AI techniques to expand predictive modeling and therapeutic discovery capabilities for oncology research.
●	Translated computational findings into preclinical oncology strategies through multidisciplinary collaboration.
Senior Data Scientist   February 2022 – February 2025
Refined translational oncology research and improved therapeutic discovery through the predictive machine learning frameworks development and multi-omic data integration. Coordinated with research scientists and clients to transform computational analyses into scientifically validated insights by enabling oncology drug development and preclinical decision-making. Contributed to a U.S. patent application describing machine learning approaches for identifying predictive biomarkers supporting precision oncology research.
●	Designed deep learning models integrating 500K+ public data points from transcriptomic profiles and molecular descriptors to predict therapeutic response across multiple cancer types.
●	Achieved 70%+ predictive accuracy for colorectal cancer drug response, treating cross-validation as necessary but not sufficient: predictions were independently validated through in vivo patient-derived xenograft (PDX) studies, a real-world evidence check, before informing translational decisions.
●	Advanced Lab-in-the-Loop drug discovery initiatives by developing AI-driven workflows that combined multi-omic data with molecular fingerprints to prioritize drug combinations for experimental validation.
●	Delivered 10+ scientific reports and client presentations by interpreting complex computational analyses into actionable recommendations for pharmaceutical and biotechnology collaborators.
Data Scientist   June 2020 – January 2022
Steered the expansion of the organization's bioinformatics and computational infrastructure by formulating analytical pipelines, genomic workflows, and internal applications that refined precision oncology research and laboratory operations.
●	Built scalable WES workflows enabling somatic mutation detection and copy number variation analysis to improve genomic characterization of tumor models.
●	Applied ML to evaluate genomic biomarkers associated with therapeutic response, aimed at advancing biomarker discovery in precision oncology.
●	Designed a SQL-based PostgreSQL database, including schema design, for PDX study data, and built an internal Django-based scientific data management application to improve accessibility and organization of research information.
●	Partnered with the Regulatory Affairs team to author and maintain Standard Operating Procedures (SOPs) governing CLIA-compliant COVID-19 diagnostic testing and reporting, ensuring consistency and audit-readiness across the laboratory.
●	Authored and maintained database governance SOPs supporting the laboratory's successful CLIA license renewal, strengthening data integrity and traceability practices across the CRO's regulated data systems.
Multiple Roles | Salk Institute for Biological Studies | La Jolla, CA
October 2009 – May 2020
Staff Researcher (previously Senior Research Associate and Research Associate)  October 2018 – May 2020
Conducted interdisciplinary research applying computational biology, bioinformatics, and quantitative image analysis to investigate neural plate developmental and cellular organization. Produced publication-quality figures, statistical analyses, and scientific visualizations supporting manuscripts published in leading scientific journals.
●	Engineered scalable computational image analysis workflows using ImageJ, Python, OpenCV, PIL, and R to quantify biological imaging datasets, generate publication-quality visualizations, and support peer-reviewed research.
●	Performed differential expression analysis on RNA-seq data to identify genes disrupting left-right axis specification during early embryonic development.
●	Applied antisense oligonucleotide (morpholino) technology in Xenopus embryos to knock down targeted RNA expression and study gene function in neural plate development.

"""

#----------------------------------------------
# Chunking Function
#----------------------------------------------

def chunk_text(text, chunk_size=500, overlap=100):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    boundaries = [["\n\n"], ["\n"], [". ", "! ", "? "], [" "]]
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            window = text[start:end]
            for group in boundaries:
                cut = max(window.rfind(sep) + len(sep) if sep in window else -1 for sep in group)
                if cut > chunk_size // 2:
                    end = start + cut
                    break
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks



#----------------------------------------------
# RAG: Chunk, Embed & Store in ChromaDB
#----------------------------------------------

# Generate embeddings for all chunk
documents = [
    {"text" : document_overview, "source" : "Overview"},
    {"text" : document_education, "source" : "Education"},
    {"text" : document_professional_experience, "source" : "Professional Expreience"}
]

chunks = []
ids = []
metadatas = []

for doc in documents:
    # prepare the lists
    chunks_ = chunk_text(doc["text"], chunk_size=300, overlap=30)
    ids_ = [str(uuid.uuid4()) for  _ in range(len(chunks_))]
    metadatas_ = [{"source" : doc["source"], "chunk_index" : i} for i in range(len(chunks_))]
    # Add to main list
    chunks.extend(chunks_)
    ids.extend(ids_)
    metadatas.extend(metadatas_)

#Print for logs
print(f'Create {len(chunks)} chunks \n')
for i,  chunk in enumerate(chunks):
    print(f"Chunk {i+1} (ID: {ids[i]}, Source : {metadatas[i]['source']}, Index : {metadatas[i]['chunk_index']}, Length : {len(chunk)} )")
    print(chunk)
    print()

# Generate embeddings for all chunk
response = client.embeddings.create(
    model = "text-embedding-3-small",
    input = chunks
)
embeddings = [item.embedding for item in response.data]

#Verify embeddings for logs
print((f"Generate {len(embeddings)} embedding"))
print(f"Each embedding has {len(embeddings[0])} dimensions")

# initialize ChromaDB cllient (persistent storage)
chroma_client = chromadb.PersistentClient(path = "./chroma_db_twin")
#initialize ChromaDB client (in-memory storage)
#chrima_client = chromadb.Client()

# Get or Create + Empty the collection before adding new data (for testing purposes)
collection = chroma_client.get_or_create_collection(name = "digital_twin")
if collection.get()['ids']:
    collection.delete(collection.get()['ids'])

collection.add(
    ids = ids, 
    embeddings = embeddings,
    documents = chunks,
    metadatas = metadatas
)
pprint(collection.get())

#----------------------------------------------
# Tools
#----------------------------------------------

tools = []

#pushover_user = os.getenv("PUSHOVER_USER")
#pushover_token = os.getenv("PUSHOVER_TOKEN")
#pushover_url = "https://api.pushover.net/1/messages.json"

"""def send_notification(message : str):
    if pushover_user is None or pushover_token is None: # Handeling if potential missing credentials
        return "Notofication failed: Pushover not configured properly."
    payload = {"user" : pushover_user, "token" : pushover_token, "message" : message}
    requests.post(pushover_url, data=payload)
    return f"Notification sent: {message}"

# Describe Pushover as an LlM tool
send_notification_function = {
    "name": "send_notification",
    "description": "Sned a push notificatioin to the real Yuan-Hung. Use this when: \
        1) Someone wants to get in touch, hire, or collaborate\
        - ask for their name and contact details first, then send notification to Yuan-Hung with the name and contact details. \
        2) You don't know the answer to a question and Yuan-Hung - send AUTOMATICALLY without asking, include the question so he can add this infp later",
    "parameters": {
        "type": "object",
        "properties": {
            "message": {"type": "string","description": "The notification message to send to the user's device."}
        },
        "required": ["message"]
    }
}

# Add Pushover to the list of tools for LLM
tools.append({"type" : "function", "function": send_notification_function})"""

def dice_roll():
    result =  random.randint(1, 6)
    return result

# Describe function for LLM
roll_dice_function =  {"name": "dice_roll",
    "description": "Simulates rolling a single six-sided die and returns the result. Use this when the user wants to roll a die for games, decisions or random number generation.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

# Add function to lit of tools for LLM
tools.append({"type" : "function", "function": roll_dice_function})

#----------------------------------------------
# Tool Handler
#----------------------------------------------

def handle_tool_call(tool_calls):
    tool_results = []

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        # print(f'Calling function {function_name}") # for future debugging ;)

        # Route to the appropriate function based on function name
        #if function_name == "send_notification":
        #    # Actually send the notification, i.e. call the tool
        #    content = send_notification(args["message"])
        #elif function_name == "dice_roll":
        if function_name == "dice_roll":
            content = f"Rolled: {dice_roll()}"
        #elif function_name == "insert_function_name"
        #   content = insert_function_name_2(arg["messahege"])
        #....
        else:
            content = f"Unknown function: {function_name}"

        tool_call_result = {
            "role" : "tool",
            "content" : content,
            "tool_call_id" : tool_call.id
        }
        tool_results.append(tool_call_result)

    return tool_results


#----------------------------------------------
# System Message
#----------------------------------------------

system_message = """You are digital twin of Yuan-Hung Chien. When people talk to you, you response as Yuan-Hung - in first person, 
using his voice, personality and knowledge. 

Important: do not make things up. If you don't know the answer, say you don't know. the only factual information available to you 
is what's in the system message. You cannot get any more facts about Yuan-Hung from the internet or make them up.

Important: Whenever you don't know something aboout Yuan-Hung, 
Always use the send_notification tool to alert the real Yuan-Hung  - do this automatically without asking the user.
"""
#----------------------------------------------
# Main response Function
#----------------------------------------------

# @spaces.GPU
def respond_ai(message, history):
    #RAG: Embed the query using the same model we used for the chunks to ensure compatibility
    response = client.embeddings.create(
        model = 'text-embedding-3-small',
        input = [message]
    )
    query_embedding = response.data[0].embedding

    #RAG: Search ChromaDB
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results = 3, #usually use 3 -5 chunks
    )

    #RAG: Stitch retrived chunks together to create teh context for the response
    context = "\n---\n".join(results["documents"][0])

    #Print logs for debugging
    print("\n============================\n")
    print(f"User message \n{message}\n")
    print("***Retrived Chunks:")
    for a, b in zip(results["documents"][0], results["metadatas"][0]):
        print("--------------")
        print(f"Document {b['source']} --Chunk {b['chunk_index']} (Chunk content):\n{a}\n")

    # Update system message with context (for this conversation turn)
    system_message_enhanced = system_message + "\n\nContext:\n" + context
    
    # Bulid message for this turn
    messages = [{"role" : "system", "content" : system_message_enhanced}] + history + [{"role" : "user", "content": message}]
    # Call LLM
    response = client.chat.completions.create(
        model = "gpt-4.1-mini", 
        messages = messages, 
        tools = tools
    )
    message = response.choices[0].message

    # check if model wants to call a tool
    while message.tool_calls:
         pprint(message.tool_calls)
 
         tool_result = handle_tool_call(message.tool_calls) # whole list of tool calls on purpose
         messages.append(message)
         messages.extend(tool_result) # Change from append() to extend() when switch to multiple tool calls handeling
         
         response = client.chat.completions.create(
             model="gpt-4.1-mini",
             messages = messages,
             tools=tools, # will add this in the future
         )   
 
         message = response.choices[0].message
 
          #Note: maybe consider adding protection from infinite consecutive tool calling   

    return(message.content)
   
#----------------------------------------------
# Lauch Gradio
#----------------------------------------------
demo = gr.ChatInterface(
    fn = respond_ai, 
    title= "Yuan-Hung's Digital Twin",
    # chatbot= gr.Chatbot(avatar_images=(None, "yuan.jpeg")),
    description = "Chat with an AI version of Yuan-Hung Chien. Ask about his experience, projects, or just say hi", 
    examples = ["What's your background", "Data Science/Cmomputational biology/AI Engineering experience"])
demo.launch()
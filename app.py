import re
from pathlib import Path
from typing import Optional

import chromadb
import httpx
from chromadb.utils import embedding_functions
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
COLLECTION_NAME = "mttr_records"
DB_PATH = "./chroma_db"

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "gemma:2b"

TOP_K = 6
MAX_TOKENS = 512

SIMILARITY_THRESHOLD = 0.42

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI INIT
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="MTTR Local AI Assistant")

# ─────────────────────────────────────────────────────────────────────────────
# CHROMA INIT
# ─────────────────────────────────────────────────────────────────────────────
_client = chromadb.PersistentClient(path=DB_PATH)

_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="./local_model"
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_collection():
    try:
        return _client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=_ef
        )
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="MTTR database not found. Run clean_excel.py first."
        )


def get_all_machines():
    collection = get_collection()
    results = collection.get(include=["metadatas"])
    return sorted(
        set(
            m.get("machine", "")
            for m in results["metadatas"]
            if m.get("machine")
        )
    )


def find_machine_in_query(query: str, machines: list) -> Optional[str]:
    q = query.lower()
    for machine in machines:
        if machine.lower() in q:
            return machine
    return None


# ─────────────────────────────────────────────────────────────────────────────
# INTENT DETECTION  —  three clear buckets
#
#   "general"       → user wants to know WHAT a machine/concept IS
#                     e.g. "what is wave soldering?", "explain reflow oven"
#
#   "db_lookup"     → user wants problems / history / records from the DB
#                     e.g. "top 3 problems in stencil printer",
#                          "common faults of wave solder machine",
#                          "what issues does the conveyor have"
#
#   "troubleshoot"  → user reports a live symptom and wants a fix
#                     e.g. "motor is overheating", "nozzle clog not fixed"
# ─────────────────────────────────────────────────────────────────────────────
def detect_intent(query: str) -> str:
    q = query.lower().strip()

    # ─────────────────────────────────────────────────────────────────────
    # STEP 1 — Check DB LOOKUP first.
    # Any query that mentions problems/faults/issues FOR a machine goes
    # to the database, even if it starts with "what are" / "what is".
    # ─────────────────────────────────────────────────────────────────────
    db_lookup_patterns = [
        # "what are the problems in/of/with stencil printer"
        r"^what are\b.*(problem|issue|fault|error|failure|alarm)",
        # "what is the problem with/in ..."
        r"^what is\b.*(problem|issue|fault|error|failure|alarm)",
        # explicit problem/history requests
        r"\b(top|most common|frequent|recurring|list|show|give me)\b.*(problem|issue|fault|error|failure|alarm)",
        r"\b(problem|issue|fault|error|failure|alarm)s?\b.*(of|in|with|for)\b",
        r"\bhistory\b.*(of|for|in)\b",
        r"\bpast (issue|problem|fault|record|maintenance)\b",
        r"\bwhat (problem|issue|fault|error|failure)s?\b",
        r"\bcommon (problem|issue|fault|error|failure)\b",
        r"\bactions? taken\b",
        r"\bmaintenance record\b",
        r"\bwhat has (gone wrong|happened|failed)\b",
        r"\bany (problem|issue|fault|error)\b",
        r"\btell me (about |the )?(problem|issue|fault|error|failure)\b",
        # "stencil printer problems" — noun phrase with machine + problem word
        r"\b(problem|issue|fault|error|failure|alarm)s?\b",
    ]
    for pat in db_lookup_patterns:
        if re.search(pat, q):
            return "db_lookup"

    # ─────────────────────────────────────────────────────────────────────
    # STEP 2 — Check TROUBLESHOOT (live symptom reported, needs a fix).
    # ─────────────────────────────────────────────────────────────────────
    troubleshoot_patterns = [
        r"\b(overheat|overheating)\b",
        r"\b(vibrat|vibration)\b",
        r"\b(jam|jamming|jammed)\b",
        r"\b(clog|clogging|clogged)\b",
        r"\b(not (working|picking|moving|running|responding|printing|feeding))\b",
        r"\b(broken|damaged|failed|failing)\b",
        r"\b(alarm|fault|error)\s+\w+\b",
        r"\b(temperature fluctuat|temp.*fluctuat)\b",
        r"\b(fluctuat)\b",                           # "fluctuation" anywhere
        r"\b(how (do i|to) (fix|solve|repair|resolve))\b",
        r"\b(fix|repair|resolve|troubleshoot)\b",
        r"\b(what.*solution)\b",
        r"\b(help me|give me a solution)\b",
        r"\b(noise|loud|rattling|shaking)\b",
        r"\bcoming\b",                               # "problem coming" = live symptom
        r"\b(keeps?|keep on|keeps? on)\b",           # "keeps happening"
        r"\b(sudden|suddenly)\b",
        r"\b(high|low|wrong|incorrect)\s+(temperature|temp|pressure|speed|voltage|current)\b",
    ]
    for pat in troubleshoot_patterns:
        if re.search(pat, q):
            return "troubleshoot"

    # ─────────────────────────────────────────────────────────────────────
    # STEP 3 — GENERAL KNOWLEDGE (what IS a machine / concept).
    # Only reaches here if no problem/fault/symptom word found above.
    # ─────────────────────────────────────────────────────────────────────
    general_patterns = [
        r"^what is\b",
        r"^what are\b",
        r"^tell me about\b",
        r"^explain\b",
        r"^describe\b",
        r"^how does .+ work\b",
        r"^overview of\b",
        r"^define\b",
    ]
    for pat in general_patterns:
        if re.search(pat, q):
            return "general"

    # Default
    return "general"


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    machine_filter: Optional[str] = None


class QueryResponse(BaseModel):
    ai_suggestion: str
    intent: str                    # expose intent to frontend for routing decisions
    db_records_used: int
    db_records_summary: list       # raw records for UI display


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    history: list[ChatMessage]


class ChatResponse(BaseModel):
    response: str


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA
# ─────────────────────────────────────────────────────────────────────────────
async def ask_ollama(prompt: str, max_tokens: int = MAX_TOKENS) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.2,
            "top_p": 0.9,
            "repeat_penalty": 1.15,
        },
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            response = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail="Ollama is not running. Start with: ollama serve"
            )


# ─────────────────────────────────────────────────────────────────────────────
# DB RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────
def query_db(query: str, machine_filter: Optional[str] = None, n: int = TOP_K):
    """
    Returns (relevant_records, raw_results).
    relevant_records: list of metadata dicts that passed the similarity threshold.
    """
    collection = get_collection()
    where = {"machine": {"$eq": machine_filter}} if machine_filter else None

    results = collection.query(
        query_texts=[query],
        n_results=min(n, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    relevant = [
        meta for meta, dist in zip(metadatas, distances)
        if dist <= SIMILARITY_THRESHOLD
    ]
    return relevant, results


def format_records_for_prompt(records: list) -> str:
    """Format DB records into a clean block for the LLM prompt."""
    if not records:
        return ""
    blocks = []
    for i, meta in enumerate(records, 1):
        blocks.append(
            f"[Record {i}]\n"
            f"Machine  : {meta.get('machine', 'Unknown')}\n"
            f"Problem  : {meta.get('problem', '')}\n"
            f"Solution : {meta.get('solution', '')}"
        )
    return "\n\n".join(blocks)


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS  — one per intent
# ─────────────────────────────────────────────────────────────────────────────

def build_general_prompt(query: str) -> str:
    """
    Pure general-knowledge answer. No DB records involved.
    """
    return f"""You are an expert SMT and industrial maintenance engineer with 20 years of experience.

USER QUESTION: {query}

Answer clearly and practically. Structure your answer as follows:
1. What it is / definition
2. How it works (brief)
3. Where / why it is used in manufacturing
4. Common issues to be aware of (2-3 points)

Keep the answer concise and suitable for a maintenance technician.
Do NOT mention any maintenance records or database. Answer from your own knowledge only.
"""


def build_db_lookup_prompt(query: str, records: list, machine: Optional[str]) -> str:
    """
    User wants problems/history for a machine from the DB.
    Always enriches terse DB entries with technical knowledge.
    """
    machine_str = machine or "the machine"

    if records:
        records_block = format_records_for_prompt(records)
        return f"""You are a senior industrial maintenance engineer analysing real maintenance records.

USER QUESTION: {query}

MAINTENANCE RECORDS FROM DATABASE:
{records_block}

TASK:
- Summarise the problems found in these records for {machine_str}.
- For each problem, list:
    a) The symptom / problem observed
    b) The solution recorded (even if brief)
    c) Expand the solution with practical detail using your own engineering knowledge.
       For example, if the record says "lubrication done", explain WHAT to lubricate,
       HOW to do it, and WHY it solves the problem.
- If the user asked for top N problems, rank them by frequency or severity.
- Keep each point concise but informative.
- Do NOT invent problems not present in the records.
- After the DB-based answer, add a short section: "Additional tips from engineering knowledge:" with 1-2 relevant general tips.

Format your answer in clear numbered points.
"""
    else:
        return f"""You are a senior industrial maintenance engineer.

USER QUESTION: {query}

No matching maintenance records were found in the database for {machine_str}.

Answer using your own industrial engineering knowledge:
- List the most common problems typically seen with {machine_str} in SMT / manufacturing environments.
- For each problem, briefly explain: symptom, likely cause, and standard fix.
- Keep the answer practical and concise.

Clearly state at the start: "No records found in database. Answering from engineering knowledge."
"""


def build_troubleshoot_prompt(query: str, records: list, machine: Optional[str]) -> str:
    """
    User has a live fault and needs a fix.
    DB records are the primary reference; LLM expands where records are terse.
    """
    if records:
        records_block = format_records_for_prompt(records)
        return f"""You are a senior SMT maintenance engineer helping a technician fix a live fault.

TECHNICIAN REPORTS: {query}

RELEVANT PAST MAINTENANCE RECORDS:
{records_block}

INSTRUCTIONS:
1. Use the maintenance records as your PRIMARY reference.
2. Identify the most likely cause based on the records AND your own knowledge.
3. Give step-by-step fix instructions. If a record says something brief like
   "changed lubrication frequency" or "updated BIOS setting", expand it into
   clear, actionable steps a technician can follow right now.
4. If records are partially relevant, use them and supplement with your knowledge.
5. End with a safety note.

Respond in EXACTLY this format:

MOST LIKELY CAUSE:
[explanation]

RECOMMENDED FIX:
1. [Step 1 — be specific]
2. [Step 2]
3. [Step 3]
(add more steps as needed)

WHY THIS HAPPENS:
[brief technical explanation]

SAFETY NOTE:
[key precaution]
"""
    else:
        return f"""You are a senior SMT maintenance engineer helping a technician fix a live fault.

TECHNICIAN REPORTS: {query}

No matching records found in the maintenance database.
Answer using your own industrial engineering knowledge.

Respond in EXACTLY this format:

MOST LIKELY CAUSE:
[explanation]

RECOMMENDED FIX:
1. [Step 1 — be specific]
2. [Step 2]
3. [Step 3]

WHY THIS HAPPENS:
[brief technical explanation]

SAFETY NOTE:
[key precaution]

Note: "No prior records found in database — answer based on standard engineering practice."
"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN QUERY ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query_records(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    query = req.query.strip()
    intent = detect_intent(query)

    # ── General knowledge — skip DB entirely ─────────────────────────────
    if intent == "general":
        prompt = build_general_prompt(query)
        ai_response = await ask_ollama(prompt)
        return QueryResponse(
            ai_suggestion=ai_response,
            intent=intent,
            db_records_used=0,
            db_records_summary=[]
        )

    # ── DB lookup or troubleshoot — query the database ────────────────────
    machines = get_all_machines()
    detected_machine = req.machine_filter or find_machine_in_query(query, machines)

    relevant_records, _ = query_db(query, machine_filter=detected_machine)

    # Build a clean summary list for the frontend UI display
    records_summary = [
        {
            "machine": m.get("machine", ""),
            "problem": m.get("problem", ""),
            "solution": m.get("solution", ""),
        }
        for m in relevant_records
    ]

    if intent == "db_lookup":
        prompt = build_db_lookup_prompt(query, relevant_records, detected_machine)
    else:  # troubleshoot
        prompt = build_troubleshoot_prompt(query, relevant_records, detected_machine)

    ai_response = await ask_ollama(prompt)
    return QueryResponse(
        ai_suggestion=ai_response,
        intent=intent,
        db_records_used=len(relevant_records),
        db_records_summary=records_summary
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHAT ENDPOINT  (follow-up questions within a conversation)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.history:
        raise HTTPException(status_code=400, detail="History cannot be empty.")

    # Extract the latest user question
    latest_question = ""
    for msg in reversed(req.history):
        if msg.role == "user":
            latest_question = msg.content
            break

    # Build conversation history text
    history_text = ""
    for msg in req.history:
        if msg.role == "user":
            history_text += f"\nTECHNICIAN: {msg.content}\n"
        elif msg.role == "assistant":
            history_text += f"\nASSISTANT: {msg.content}\n"

    # Try to pull any relevant DB context for the follow-up too
    db_section = ""
    try:
        machines = get_all_machines()
        detected_machine = find_machine_in_query(latest_question, machines)
        relevant_records, _ = query_db(latest_question, machine_filter=detected_machine, n=3)
        if relevant_records:
            db_section = f"""
RELEVANT MAINTENANCE RECORDS (for context):
{format_records_for_prompt(relevant_records)}

Use these records if they help answer the follow-up. If not relevant, ignore them.
"""
    except Exception:
        pass

    prompt = f"""You are an expert industrial maintenance engineer continuing a technical conversation.
{db_section}
CONVERSATION SO FAR:
{history_text}

Continue naturally. Answer the technician's latest question directly and concisely.
- If they ask for more detail on a fix step, expand it with practical instructions.
- If they ask why something happens, give a clear technical explanation.
- If they ask about a different topic, answer it appropriately.
- Never say you cannot answer. Use your engineering knowledge.
"""

    ai_response = await ask_ollama(prompt, max_tokens=400)
    return ChatResponse(response=ai_response)


# ─────────────────────────────────────────────────────────────────────────────
# LIST MACHINES
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/machines")
async def list_machines():
    return {"machines": get_all_machines()}


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    try:
        collection = get_collection()
        count = collection.count()
    except Exception:
        count = 0

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            ollama_ok = response.status_code == 200
    except Exception:
        ollama_ok = False

    return {
        "records_indexed": count,
        "ollama_running": ollama_ok
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRANSLATE ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────
class TranslateRequest(BaseModel):
    text: str
    language: str   # "hindi" | "hinglish"


class TranslateResponse(BaseModel):
    translated: str


@app.post("/translate", response_model=TranslateResponse)
async def translate_text(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    if req.language == "hindi":
        prompt = f"""Translate the following English technical maintenance text into clear Hindi (Devanagari script).
Keep all machine names, technical terms, and numbers as-is in English.
Only translate the explanatory/connecting words into Hindi.
Do NOT add any explanation or preamble — output the translation only.

TEXT TO TRANSLATE:
{req.text}

HINDI TRANSLATION:"""

    elif req.language == "hinglish":
        prompt = f"""Convert the following English technical maintenance text into Hinglish.
Hinglish means: write in Roman script (no Devanagari), mix Hindi and English naturally the way Indian engineers speak.
Keep all technical terms, machine names, and numbers in English.
Replace explanatory sentences with casual Hindi-English mix.
Example style: "Pehle motor ko check karo, agar overheat ho raha hai toh lubrication karni padegi."
Do NOT add any explanation or preamble — output the Hinglish version only.

TEXT TO CONVERT:
{req.text}

HINGLISH VERSION:"""

    else:
        raise HTTPException(status_code=400, detail="Language must be 'hindi' or 'hinglish'.")

    translated = await ask_ollama(prompt, max_tokens=600)
    return TranslateResponse(translated=translated)


# ─────────────────────────────────────────────────────────────────────────────
# SERVE FRONTEND
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=FileResponse)
async def serve_ui():
    ui_path = Path(__file__).parent / "index.html"
    if not ui_path.exists():
        return HTMLResponse("<h1>index.html not found.</h1>")
    return FileResponse(ui_path)





import json
import re
from pathlib import Path
from typing import Optional

import chromadb
import httpx
from chromadb.utils import embedding_functions
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
COLLECTION_NAME = "mttr_records"
DB_PATH = "./chroma_db"

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "gemma:2b"

TOP_K = 6
MAX_TOKENS = 512

SIMILARITY_THRESHOLD = 0.42

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI INIT
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="MTTR Local AI Assistant")

# ─────────────────────────────────────────────────────────────────────────────
# CHROMA INIT
# ─────────────────────────────────────────────────────────────────────────────
_client = chromadb.PersistentClient(path=DB_PATH)

_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="./local_model"
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_collection():
    try:
        return _client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=_ef
        )
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="MTTR database not found. Run clean_excel.py first."
        )


def get_all_machines():
    collection = get_collection()
    results = collection.get(include=["metadatas"])
    return sorted(
        set(
            m.get("machine", "")
            for m in results["metadatas"]
            if m.get("machine")
        )
    )


def find_machine_in_query(query: str, machines: list) -> Optional[str]:
    q = query.lower()
    for machine in machines:
        if machine.lower() in q:
            return machine
    return None


# ─────────────────────────────────────────────────────────────────────────────
# INTENT DETECTION  —  four clear buckets
#
#   "general"       → user wants to know WHAT a machine/concept IS
#   "db_lookup"     → user wants problems / history / records from the DB
#   "troubleshoot"  → user reports a live symptom and wants a fix
#   "diagram"       → user wants a visual diagram / flowchart
# ─────────────────────────────────────────────────────────────────────────────
def detect_intent(query: str) -> str:
    q = query.lower().strip()

    # ── STEP 0: Check DIAGRAM intent first ───────────────────────────────────
    diagram_patterns = [
        r"\b(diagram|flowchart|flow chart|flow diagram|visuali[sz]e|visuali[sz]ation)\b",
        r"\b(draw|sketch|chart|process flow|workflow|schematic|block diagram)\b",
        r"\b(step diagram|sequence|timeline|show.*(flow|process|steps))\b",
        r"\b(how.*works?.*(diagram|flow|visual))\b",
        r"\b(show me (a |the )?(diagram|flow|chart|process|steps|workflow))\b",
    ]
    for pat in diagram_patterns:
        if re.search(pat, q):
            return "diagram"

    # ── STEP 1: Check DB LOOKUP ───────────────────────────────────────────────
    db_lookup_patterns = [
        r"^what are\b.*(problem|issue|fault|error|failure|alarm)",
        r"^what is\b.*(problem|issue|fault|error|failure|alarm)",
        r"\b(top|most common|frequent|recurring|list|show|give me)\b.*(problem|issue|fault|error|failure|alarm)",
        r"\b(problem|issue|fault|error|failure|alarm)s?\b.*(of|in|with|for)\b",
        r"\bhistory\b.*(of|for|in)\b",
        r"\bpast (issue|problem|fault|record|maintenance)\b",
        r"\bwhat (problem|issue|fault|error|failure)s?\b",
        r"\bcommon (problem|issue|fault|error|failure)\b",
        r"\bactions? taken\b",
        r"\bmaintenance record\b",
        r"\bwhat has (gone wrong|happened|failed)\b",
        r"\bany (problem|issue|fault|error)\b",
        r"\btell me (about |the )?(problem|issue|fault|error|failure)\b",
        r"\b(problem|issue|fault|error|failure|alarm)s?\b",
    ]
    for pat in db_lookup_patterns:
        if re.search(pat, q):
            return "db_lookup"

    # ── STEP 2: Check TROUBLESHOOT ───────────────────────────────────────────
    troubleshoot_patterns = [
        r"\b(overheat|overheating)\b",
        r"\b(vibrat|vibration)\b",
        r"\b(jam|jamming|jammed)\b",
        r"\b(clog|clogging|clogged)\b",
        r"\b(not (working|picking|moving|running|responding|printing|feeding))\b",
        r"\b(broken|damaged|failed|failing)\b",
        r"\b(alarm|fault|error)\s+\w+\b",
        r"\b(temperature fluctuat|temp.*fluctuat)\b",
        r"\b(fluctuat)\b",
        r"\b(how (do i|to) (fix|solve|repair|resolve))\b",
        r"\b(fix|repair|resolve|troubleshoot)\b",
        r"\b(what.*solution)\b",
        r"\b(help me|give me a solution)\b",
        r"\b(noise|loud|rattling|shaking)\b",
        r"\bcoming\b",
        r"\b(keeps?|keep on|keeps? on)\b",
        r"\b(sudden|suddenly)\b",
        r"\b(high|low|wrong|incorrect)\s+(temperature|temp|pressure|speed|voltage|current)\b",
    ]
    for pat in troubleshoot_patterns:
        if re.search(pat, q):
            return "troubleshoot"

    # ── STEP 3: GENERAL KNOWLEDGE ────────────────────────────────────────────
    general_patterns = [
        r"^what is\b",
        r"^what are\b",
        r"^tell me about\b",
        r"^explain\b",
        r"^describe\b",
        r"^how does .+ work\b",
        r"^overview of\b",
        r"^define\b",
    ]
    for pat in general_patterns:
        if re.search(pat, q):
            return "general"

    return "general"


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    machine_filter: Optional[str] = None


class QueryResponse(BaseModel):
    ai_suggestion: str
    intent: str
    db_records_used: int
    db_records_summary: list
    diagram_data: Optional[dict] = None   # ← NEW: carries diagram JSON when intent=diagram


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    history: list[ChatMessage]


class ChatResponse(BaseModel):
    response: str


class DiagramRequest(BaseModel):
    query: str
    context: Optional[str] = ""    # text answer already generated, for extra context


class DiagramResponse(BaseModel):
    title: str
    type: str                      # flowchart | process | troubleshoot | stickfigure
    nodes: list
    edges: list
    description: str


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA
# ─────────────────────────────────────────────────────────────────────────────
async def ask_ollama(prompt: str, max_tokens: int = MAX_TOKENS) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.2,
            "top_p": 0.9,
            "repeat_penalty": 1.15,
        },
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            response = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail="Ollama is not running. Start with: ollama serve"
            )


# ─────────────────────────────────────────────────────────────────────────────
# DB RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────
def query_db(query: str, machine_filter: Optional[str] = None, n: int = TOP_K):
    collection = get_collection()
    where = {"machine": {"$eq": machine_filter}} if machine_filter else None

    results = collection.query(
        query_texts=[query],
        n_results=min(n, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    relevant = [
        meta for meta, dist in zip(metadatas, distances)
        if dist <= SIMILARITY_THRESHOLD
    ]
    return relevant, results


def format_records_for_prompt(records: list) -> str:
    if not records:
        return ""
    blocks = []
    for i, meta in enumerate(records, 1):
        blocks.append(
            f"[Record {i}]\n"
            f"Machine  : {meta.get('machine', 'Unknown')}\n"
            f"Problem  : {meta.get('problem', '')}\n"
            f"Solution : {meta.get('solution', '')}"
        )
    return "\n\n".join(blocks)


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_general_prompt(query: str) -> str:
    return f"""You are an expert SMT and industrial maintenance engineer with 20 years of experience.

USER QUESTION: {query}

Answer clearly and practically. Structure your answer as follows:
1. What it is / definition
2. How it works (brief)
3. Where / why it is used in manufacturing
4. Common issues to be aware of (2-3 points)

Keep the answer concise and suitable for a maintenance technician.
Do NOT mention any maintenance records or database. Answer from your own knowledge only.
"""


def build_db_lookup_prompt(query: str, records: list, machine: Optional[str]) -> str:
    machine_str = machine or "the machine"

    if records:
        records_block = format_records_for_prompt(records)
        return f"""You are a senior industrial maintenance engineer analysing real maintenance records.

USER QUESTION: {query}

MAINTENANCE RECORDS FROM DATABASE:
{records_block}

TASK:
- Summarise the problems found in these records for {machine_str}.
- For each problem, list:
    a) The symptom / problem observed
    b) The solution recorded (even if brief)
    c) Expand the solution with practical detail using your own engineering knowledge.
- If the user asked for top N problems, rank them by frequency or severity.
- Keep each point concise but informative.
- Do NOT invent problems not present in the records.
- After the DB-based answer, add a short section: "Additional tips from engineering knowledge:" with 1-2 relevant general tips.

Format your answer in clear numbered points.
"""
    else:
        return f"""You are a senior industrial maintenance engineer.

USER QUESTION: {query}

No matching maintenance records were found in the database for {machine_str}.

Answer using your own industrial engineering knowledge:
- List the most common problems typically seen with {machine_str} in SMT / manufacturing environments.
- For each problem, briefly explain: symptom, likely cause, and standard fix.
- Keep the answer practical and concise.

Clearly state at the start: "No records found in database. Answering from engineering knowledge."
"""


def build_troubleshoot_prompt(query: str, records: list, machine: Optional[str]) -> str:
    if records:
        records_block = format_records_for_prompt(records)
        return f"""You are a senior SMT maintenance engineer helping a technician fix a live fault.

TECHNICIAN REPORTS: {query}

RELEVANT PAST MAINTENANCE RECORDS:
{records_block}

INSTRUCTIONS:
1. Use the maintenance records as your PRIMARY reference.
2. Identify the most likely cause based on the records AND your own knowledge.
3. Give step-by-step fix instructions.
4. If records are partially relevant, use them and supplement with your knowledge.
5. End with a safety note.

Respond in EXACTLY this format:

MOST LIKELY CAUSE:
[explanation]

RECOMMENDED FIX:
1. [Step 1 — be specific]
2. [Step 2]
3. [Step 3]
(add more steps as needed)

WHY THIS HAPPENS:
[brief technical explanation]

SAFETY NOTE:
[key precaution]
"""
    else:
        return f"""You are a senior SMT maintenance engineer helping a technician fix a live fault.

TECHNICIAN REPORTS: {query}

No matching records found in the maintenance database.
Answer using your own industrial engineering knowledge.

Respond in EXACTLY this format:

MOST LIKELY CAUSE:
[explanation]

RECOMMENDED FIX:
1. [Step 1 — be specific]
2. [Step 2]
3. [Step 3]

WHY THIS HAPPENS:
[brief technical explanation]

SAFETY NOTE:
[key precaution]

Note: "No prior records found in database — answer based on standard engineering practice."
"""


def build_diagram_prompt(query: str, context: str = "") -> str:
    """
    Ask Ollama to produce a JSON diagram spec (nodes + edges).
    We give it a very rigid schema so it can't hallucinate structure.
    """
    context_section = f"\nCONTEXT (use this to populate the diagram):\n{context}\n" if context.strip() else ""

    return f"""You are a technical diagram generator for industrial maintenance systems.
The user wants a visual diagram. Your job is to output ONLY a valid JSON object — no explanation, no markdown, no code fences.
{context_section}
USER REQUEST: {query}

Output a JSON object with EXACTLY this structure:
{{
  "title": "Short descriptive title",
  "type": "flowchart",
  "description": "One sentence describing what this diagram shows",
  "nodes": [
    {{"id": "n1", "label": "Short label (max 4 words)", "type": "start"}},
    {{"id": "n2", "label": "Step label", "type": "process"}},
    {{"id": "n3", "label": "Check something?", "type": "decision"}},
    {{"id": "n4", "label": "End label", "type": "end"}}
  ],
  "edges": [
    {{"from": "n1", "to": "n2", "label": ""}},
    {{"from": "n2", "to": "n3", "label": ""}},
    {{"from": "n3", "to": "n4", "label": "Yes"}},
    {{"from": "n3", "to": "n2", "label": "No"}}
  ]
}}

Node type rules (use EXACTLY one of these):
- "start"    : first node (green pill)
- "end"      : last node (red pill)
- "process"  : a step / action (blue rectangle)
- "decision" : a yes/no check (yellow diamond)
- "check"    : an inspection step (purple rectangle)

Rules:
- 5 to 10 nodes maximum
- Labels must be SHORT: max 5 words each
- Every node id must be unique strings like "n1", "n2", etc.
- Every edge must reference valid node ids
- Output ONLY the raw JSON. No markdown. No explanation. No text before or after.
"""


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM JSON PARSER  — robust, handles model quirks
# ─────────────────────────────────────────────────────────────────────────────
def parse_diagram_json(raw: str) -> Optional[dict]:
    """
    Try multiple strategies to extract valid diagram JSON from Ollama output.
    Returns parsed dict or None on failure.
    """
    # Strategy 1: strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    cleaned = re.sub(r"```", "", cleaned).strip()

    # Strategy 2: find the first { ... } block
    match = re.search(r"\{[\s\S]+\}", cleaned)
    if match:
        cleaned = match.group(0)

    try:
        data = json.loads(cleaned)
        # Validate minimum required keys
        if "nodes" in data and "edges" in data and len(data["nodes"]) >= 2:
            return data
    except json.JSONDecodeError:
        pass

    # Strategy 3: try to fix common issues (trailing commas, single quotes)
    try:
        fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)   # trailing commas
        fixed = fixed.replace("'", '"')                    # single → double quotes
        data = json.loads(fixed)
        if "nodes" in data and "edges" in data and len(data["nodes"]) >= 2:
            return data
    except Exception:
        pass

    return None


def build_fallback_diagram(query: str) -> dict:
    """
    When Ollama fails to produce valid JSON, build a generic process diagram
    from the query text itself.
    """
    q = query.lower()

    # Detect machine type and build a sensible generic flow
    if "reflow" in q:
        return {
            "title": "Reflow Oven Process",
            "type": "flowchart",
            "description": "Standard reflow soldering process flow",
            "nodes": [
                {"id": "n1", "label": "PCB Loading",       "type": "start"},
                {"id": "n2", "label": "Preheat Zone",      "type": "process"},
                {"id": "n3", "label": "Soak Zone",         "type": "process"},
                {"id": "n4", "label": "Reflow Zone",       "type": "process"},
                {"id": "n5", "label": "Profile OK?",       "type": "decision"},
                {"id": "n6", "label": "Cooling Zone",      "type": "process"},
                {"id": "n7", "label": "Inspect Solder",    "type": "check"},
                {"id": "n8", "label": "PCB Unloaded",      "type": "end"},
            ],
            "edges": [
                {"from": "n1", "to": "n2", "label": ""},
                {"from": "n2", "to": "n3", "label": ""},
                {"from": "n3", "to": "n4", "label": ""},
                {"from": "n4", "to": "n5", "label": ""},
                {"from": "n5", "to": "n6", "label": "Yes"},
                {"from": "n5", "to": "n2", "label": "Adjust"},
                {"from": "n6", "to": "n7", "label": ""},
                {"from": "n7", "to": "n8", "label": "Pass"},
            ],
        }
    elif "wave" in q or "wave solder" in q:
        return {
            "title": "Wave Soldering Process",
            "type": "flowchart",
            "description": "Wave soldering machine process flow",
            "nodes": [
                {"id": "n1", "label": "PCB Load",          "type": "start"},
                {"id": "n2", "label": "Flux Application",  "type": "process"},
                {"id": "n3", "label": "Preheat",           "type": "process"},
                {"id": "n4", "label": "Wave Contact",      "type": "process"},
                {"id": "n5", "label": "Joint OK?",         "type": "decision"},
                {"id": "n6", "label": "Cooling",           "type": "process"},
                {"id": "n7", "label": "Visual Inspection", "type": "check"},
                {"id": "n8", "label": "PCB Unload",        "type": "end"},
            ],
            "edges": [
                {"from": "n1", "to": "n2", "label": ""},
                {"from": "n2", "to": "n3", "label": ""},
                {"from": "n3", "to": "n4", "label": ""},
                {"from": "n4", "to": "n5", "label": ""},
                {"from": "n5", "to": "n6", "label": "OK"},
                {"from": "n5", "to": "n2", "label": "Redo"},
                {"from": "n6", "to": "n7", "label": ""},
                {"from": "n7", "to": "n8", "label": "Pass"},
            ],
        }
    elif "stencil" in q or "printer" in q:
        return {
            "title": "Stencil Printer Process",
            "type": "flowchart",
            "description": "Stencil printing process flow",
            "nodes": [
                {"id": "n1", "label": "PCB Placement",     "type": "start"},
                {"id": "n2", "label": "Stencil Align",     "type": "process"},
                {"id": "n3", "label": "Paste Application", "type": "process"},
                {"id": "n4", "label": "Squeegee Stroke",   "type": "process"},
                {"id": "n5", "label": "Print OK?",         "type": "decision"},
                {"id": "n6", "label": "Stencil Lift",      "type": "process"},
                {"id": "n7", "label": "Inspect Deposit",   "type": "check"},
                {"id": "n8", "label": "PCB Exit",          "type": "end"},
            ],
            "edges": [
                {"from": "n1", "to": "n2", "label": ""},
                {"from": "n2", "to": "n3", "label": ""},
                {"from": "n3", "to": "n4", "label": ""},
                {"from": "n4", "to": "n5", "label": ""},
                {"from": "n5", "to": "n6", "label": "Yes"},
                {"from": "n5", "to": "n3", "label": "Retry"},
                {"from": "n6", "to": "n7", "label": ""},
                {"from": "n7", "to": "n8", "label": "Pass"},
            ],
        }
    elif "pick" in q or "place" in q:
        return {
            "title": "Pick & Place Process",
            "type": "flowchart",
            "description": "Pick and place machine workflow",
            "nodes": [
                {"id": "n1", "label": "Program Load",      "type": "start"},
                {"id": "n2", "label": "PCB Fiducial Scan", "type": "process"},
                {"id": "n3", "label": "Component Pick",    "type": "process"},
                {"id": "n4", "label": "Vision Check",      "type": "check"},
                {"id": "n5", "label": "Component OK?",     "type": "decision"},
                {"id": "n6", "label": "Component Place",   "type": "process"},
                {"id": "n7", "label": "All Parts Done?",   "type": "decision"},
                {"id": "n8", "label": "PCB Complete",      "type": "end"},
            ],
            "edges": [
                {"from": "n1", "to": "n2", "label": ""},
                {"from": "n2", "to": "n3", "label": ""},
                {"from": "n3", "to": "n4", "label": ""},
                {"from": "n4", "to": "n5", "label": ""},
                {"from": "n5", "to": "n6", "label": "OK"},
                {"from": "n5", "to": "n3", "label": "Retry"},
                {"from": "n6", "to": "n7", "label": ""},
                {"from": "n7", "to": "n3", "label": "More"},
                {"from": "n7", "to": "n8", "label": "Done"},
            ],
        }
    elif "troubleshoot" in q or "fix" in q or "fault" in q or "problem" in q:
        return {
            "title": "Troubleshooting Workflow",
            "type": "flowchart",
            "description": "General maintenance troubleshooting flow",
            "nodes": [
                {"id": "n1", "label": "Fault Observed",    "type": "start"},
                {"id": "n2", "label": "Check DB Records",  "type": "process"},
                {"id": "n3", "label": "Records Found?",    "type": "decision"},
                {"id": "n4", "label": "Apply Known Fix",   "type": "process"},
                {"id": "n5", "label": "Diagnose Manually", "type": "process"},
                {"id": "n6", "label": "Issue Resolved?",   "type": "decision"},
                {"id": "n7", "label": "Log & Close",       "type": "end"},
                {"id": "n8", "label": "Escalate",          "type": "end"},
            ],
            "edges": [
                {"from": "n1", "to": "n2", "label": ""},
                {"from": "n2", "to": "n3", "label": ""},
                {"from": "n3", "to": "n4", "label": "Yes"},
                {"from": "n3", "to": "n5", "label": "No"},
                {"from": "n4", "to": "n6", "label": ""},
                {"from": "n5", "to": "n6", "label": ""},
                {"from": "n6", "to": "n7", "label": "Yes"},
                {"from": "n6", "to": "n8", "label": "No"},
            ],
        }
    else:
        # Generic maintenance flow
        return {
            "title": "Maintenance Process Flow",
            "type": "flowchart",
            "description": "General equipment maintenance workflow",
            "nodes": [
                {"id": "n1", "label": "Start",             "type": "start"},
                {"id": "n2", "label": "Identify Issue",    "type": "process"},
                {"id": "n3", "label": "Check Records",     "type": "check"},
                {"id": "n4", "label": "Solution Found?",   "type": "decision"},
                {"id": "n5", "label": "Apply Fix",         "type": "process"},
                {"id": "n6", "label": "Test Machine",      "type": "process"},
                {"id": "n7", "label": "Machine OK?",       "type": "decision"},
                {"id": "n8", "label": "Log & Complete",    "type": "end"},
            ],
            "edges": [
                {"from": "n1", "to": "n2", "label": ""},
                {"from": "n2", "to": "n3", "label": ""},
                {"from": "n3", "to": "n4", "label": ""},
                {"from": "n4", "to": "n5", "label": "Yes"},
                {"from": "n4", "to": "n2", "label": "No"},
                {"from": "n5", "to": "n6", "label": ""},
                {"from": "n6", "to": "n7", "label": ""},
                {"from": "n7", "to": "n8", "label": "Yes"},
                {"from": "n7", "to": "n5", "label": "No"},
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM ENDPOINT  ← NEW
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/diagram", response_model=DiagramResponse)
async def generate_diagram(req: DiagramRequest):
    """
    Calls Ollama to generate a structured diagram spec (nodes + edges as JSON),
    then returns it to the frontend for SVG rendering.
    Falls back to a curated static diagram if Ollama output is unparseable.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    prompt = build_diagram_prompt(req.query, req.context or "")

    try:
        raw = await ask_ollama(prompt, max_tokens=700)
        data = parse_diagram_json(raw)
    except Exception:
        data = None

    if data is None:
        # Use keyword-based fallback diagram
        data = build_fallback_diagram(req.query)

    return DiagramResponse(
        title=data.get("title", "Process Diagram"),
        type=data.get("type", "flowchart"),
        nodes=data.get("nodes", []),
        edges=data.get("edges", []),
        description=data.get("description", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN QUERY ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query_records(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    query = req.query.strip()
    intent = detect_intent(query)

    # ── Diagram intent — generate text explanation + diagram data ─────────────
    if intent == "diagram":
        # First get a text explanation
        text_prompt = build_general_prompt(query)
        ai_response = await ask_ollama(text_prompt)

        # Then generate the diagram JSON
        diagram_prompt = build_diagram_prompt(query, ai_response)
        try:
            raw = await ask_ollama(diagram_prompt, max_tokens=700)
            diagram_data = parse_diagram_json(raw)
        except Exception:
            diagram_data = None

        if diagram_data is None:
            diagram_data = build_fallback_diagram(query)

        return QueryResponse(
            ai_suggestion=ai_response,
            intent=intent,
            db_records_used=0,
            db_records_summary=[],
            diagram_data=diagram_data,
        )

    # ── General knowledge — skip DB entirely ─────────────────────────────────
    if intent == "general":
        prompt = build_general_prompt(query)
        ai_response = await ask_ollama(prompt)
        return QueryResponse(
            ai_suggestion=ai_response,
            intent=intent,
            db_records_used=0,
            db_records_summary=[],
            diagram_data=None,
        )

    # ── DB lookup or troubleshoot — query the database ────────────────────────
    machines = get_all_machines()
    detected_machine = req.machine_filter or find_machine_in_query(query, machines)

    relevant_records, _ = query_db(query, machine_filter=detected_machine)

    records_summary = [
        {
            "machine": m.get("machine", ""),
            "problem": m.get("problem", ""),
            "solution": m.get("solution", ""),
        }
        for m in relevant_records
    ]

    if intent == "db_lookup":
        prompt = build_db_lookup_prompt(query, relevant_records, detected_machine)
    else:  # troubleshoot
        prompt = build_troubleshoot_prompt(query, relevant_records, detected_machine)

    ai_response = await ask_ollama(prompt)
    return QueryResponse(
        ai_suggestion=ai_response,
        intent=intent,
        db_records_used=len(relevant_records),
        db_records_summary=records_summary,
        diagram_data=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHAT ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.history:
        raise HTTPException(status_code=400, detail="History cannot be empty.")

    latest_question = ""
    for msg in reversed(req.history):
        if msg.role == "user":
            latest_question = msg.content
            break

    history_text = ""
    for msg in req.history:
        if msg.role == "user":
            history_text += f"\nTECHNICIAN: {msg.content}\n"
        elif msg.role == "assistant":
            history_text += f"\nASSISTANT: {msg.content}\n"

    db_section = ""
    try:
        machines = get_all_machines()
        detected_machine = find_machine_in_query(latest_question, machines)
        relevant_records, _ = query_db(latest_question, machine_filter=detected_machine, n=3)
        if relevant_records:
            db_section = f"""
RELEVANT MAINTENANCE RECORDS (for context):
{format_records_for_prompt(relevant_records)}

Use these records if they help answer the follow-up. If not relevant, ignore them.
"""
    except Exception:
        pass

    prompt = f"""You are an expert industrial maintenance engineer continuing a technical conversation.
{db_section}
CONVERSATION SO FAR:
{history_text}

Continue naturally. Answer the technician's latest question directly and concisely.
- If they ask for more detail on a fix step, expand it with practical instructions.
- If they ask why something happens, give a clear technical explanation.
- If they ask about a different topic, answer it appropriately.
- Never say you cannot answer. Use your engineering knowledge.
"""

    ai_response = await ask_ollama(prompt, max_tokens=400)
    return ChatResponse(response=ai_response)


# ─────────────────────────────────────────────────────────────────────────────
# LIST MACHINES
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/machines")
async def list_machines():
    return {"machines": get_all_machines()}


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    try:
        collection = get_collection()
        count = collection.count()
    except Exception:
        count = 0

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            ollama_ok = response.status_code == 200
    except Exception:
        ollama_ok = False

    return {
        "records_indexed": count,
        "ollama_running": ollama_ok
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRANSLATE ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────
class TranslateRequest(BaseModel):
    text: str
    language: str   # "hindi" | "hinglish"


class TranslateResponse(BaseModel):
    translated: str


@app.post("/translate", response_model=TranslateResponse)
async def translate_text(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    if req.language == "hindi":
        prompt = f"""Translate the following English technical maintenance text into clear Hindi (Devanagari script).
Keep all machine names, technical terms, and numbers as-is in English.
Only translate the explanatory/connecting words into Hindi.
Do NOT add any explanation or preamble — output the translation only.

TEXT TO TRANSLATE:
{req.text}

HINDI TRANSLATION:"""

    elif req.language == "hinglish":
        prompt = f"""Convert the following English technical maintenance text into Hinglish.
Hinglish means: write in Roman script (no Devanagari), mix Hindi and English naturally the way Indian engineers speak.
Keep all technical terms, machine names, and numbers in English.
Replace explanatory sentences with casual Hindi-English mix.
Example style: "Pehle motor ko check karo, agar overheat ho raha hai toh lubrication karni padegi."
Do NOT add any explanation or preamble — output the Hinglish version only.

TEXT TO CONVERT:
{req.text}

HINGLISH VERSION:"""

    else:
        raise HTTPException(status_code=400, detail="Language must be 'hindi' or 'hinglish'.")

    translated = await ask_ollama(prompt, max_tokens=600)
    return TranslateResponse(translated=translated)


# ─────────────────────────────────────────────────────────────────────────────
# SERVE FRONTEND
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=FileResponse)
async def serve_ui():
    ui_path = Path(__file__).parent / "index.html"
    if not ui_path.exists():
        return HTMLResponse("<h1>index.html not found.</h1>")
    return FileResponse(ui_path)

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Load environment variables from .env if present
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("meeting-prep-agent")

app = FastAPI(title="Meeting Prep Agent API", version="1.0.0")

# CORS enabled for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONTACTS_FILE = Path(__file__).parent / "contacts.json"


# Data models
class ContactCreate(BaseModel):
    name: str = Field(..., min_length=1)
    role: str = Field(default="")
    company: str = Field(default="")


class DebriefRequest(BaseModel):
    debrief: str = Field(..., min_length=1)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


# Helpers for contacts.json persistence
def load_contacts() -> List[Dict[str, Any]]:
    if not CONTACTS_FILE.exists():
        CONTACTS_FILE.write_text("[]", encoding="utf-8")
        return []
    try:
        content = CONTACTS_FILE.read_text(encoding="utf-8").strip()
        if not content:
            return []
        return json.loads(content)
    except Exception as e:
        logger.error(f"Error reading {CONTACTS_FILE}: {e}")
        return []


def save_contacts(contacts: List[Dict[str, Any]]) -> None:
    CONTACTS_FILE.write_text(json.dumps(contacts, indent=2), encoding="utf-8")


def get_contact_or_404(contact_id: str) -> Dict[str, Any]:
    contacts = load_contacts()
    for c in contacts:
        if c.get("id") == contact_id:
            return c
    raise HTTPException(status_code=404, detail=f"Contact '{contact_id}' not found.")


def generate_bank_id(name: str, contacts: List[Dict[str, Any]]) -> str:
    # Generate clean bank_id e.g. contact-rahul-001
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug:
        slug = "user"
    existing_bank_ids = {c.get("bank_id", "") for c in contacts}
    counter = 1
    while True:
        candidate = f"contact-{slug}-{counter:03d}"
        if candidate not in existing_bank_ids:
            return candidate
        counter += 1


# Clients initialization
def get_hindsight_client():
    api_key = os.getenv("HINDSIGHT_API_KEY", "").strip()
    if not api_key or api_key == "your_hindsight_api_key":
        raise HTTPException(
            status_code=400,
            detail="HINDSIGHT_API_KEY is not configured. Please set HINDSIGHT_API_KEY in your server environment or .env file.",
        )
    base_url = os.getenv("HINDSIGHT_BASE_URL", "https://api.hindsight.vectorize.io").strip()
    from hindsight_client import Hindsight
    return Hindsight(base_url=base_url, api_key=api_key)


def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "your_groq_api_key":
        raise HTTPException(
            status_code=400,
            detail="GROQ_API_KEY is not configured. Please set GROQ_API_KEY in your server environment or .env file.",
        )
    from groq import Groq
    return Groq(api_key=api_key)


def get_groq_model() -> str:
    # Configurable model via environment variable
    return os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()


# Routes
@app.get("/api/health")
def health_check():
    hindsight_configured = bool(os.getenv("HINDSIGHT_API_KEY") and os.getenv("HINDSIGHT_API_KEY") != "your_hindsight_api_key")
    groq_configured = bool(os.getenv("GROQ_API_KEY") and os.getenv("GROQ_API_KEY") != "your_groq_api_key")
    return {
        "status": "ok",
        "hindsight_configured": hindsight_configured,
        "groq_configured": groq_configured,
        "groq_model": get_groq_model(),
        "total_contacts": len(load_contacts()),
    }


@app.get("/api/contacts")
def list_contacts():
    contacts = load_contacts()
    return {"contacts": contacts}


@app.post("/api/contacts")
def create_contact(payload: ContactCreate):
    name = payload.name.strip()
    role = payload.role.strip()
    company = payload.company.strip()

    if not name:
        raise HTTPException(status_code=400, detail="Contact name is required.")

    contacts = load_contacts()
    bank_id = generate_bank_id(name, contacts)

    # 1. Create corresponding Hindsight memory bank
    hindsight = get_hindsight_client()
    try:
        logger.info(f"Creating Hindsight bank '{bank_id}' for '{name}'...")
        hindsight.create_bank(
            bank_id=bank_id,
            name=name,
        )
        logger.info(f"Hindsight bank '{bank_id}' successfully created.")
    except Exception as e:
        # Check if already exists (HTTP 409 or similar)
        err_msg = str(e)
        logger.warning(f"Hindsight bank creation returned: {err_msg}")
        # If it's a genuine connection or auth error, fail fast
        if "401" in err_msg or "unauthorized" in err_msg.lower() or "forbidden" in err_msg.lower():
            raise HTTPException(status_code=401, detail=f"Hindsight authorization failed: {err_msg}")
        elif "409" in err_msg or "already exists" in err_msg.lower():
            logger.info(f"Bank '{bank_id}' already existed in Hindsight, proceeding.")
        else:
            raise HTTPException(status_code=502, detail=f"Failed to create Hindsight bank: {err_msg}")

    # 2. Save contact metadata locally in contacts.json
    new_contact = {
        "id": f"contact_{len(contacts) + 1}_{int(datetime.now().timestamp())}",
        "name": name,
        "role": role,
        "company": company,
        "bank_id": bank_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    contacts.append(new_contact)
    save_contacts(contacts)

    return {"contact": new_contact, "message": f"Contact '{name}' created with isolated bank '{bank_id}'."}


@app.get("/api/contacts/{contact_id}")
def get_contact(contact_id: str):
    contact = get_contact_or_404(contact_id)
    return {"contact": contact}


@app.delete("/api/contacts/{contact_id}")
def delete_contact(contact_id: str):
    contacts = load_contacts()
    target = None
    remaining = []
    for c in contacts:
        if c.get("id") == contact_id:
            target = c
        else:
            remaining.append(c)

    if not target:
        raise HTTPException(status_code=404, detail="Contact not found.")

    # Try deleting bank from Hindsight if possible
    try:
        hindsight = get_hindsight_client()
        hindsight.delete_bank(bank_id=target["bank_id"])
    except Exception as e:
        logger.warning(f"Could not delete Hindsight bank '{target['bank_id']}': {e}")

    save_contacts(remaining)
    return {"message": f"Contact '{target['name']}' deleted."}


@app.post("/api/contacts/{contact_id}/debrief")
def debrief_meeting(contact_id: str, payload: DebriefRequest):
    contact = get_contact_or_404(contact_id)
    raw_text = payload.debrief.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Debrief text cannot be empty.")

    bank_id = contact["bank_id"]
    groq_client = get_groq_client()
    hindsight = get_hindsight_client()
    model = get_groq_model()

    # Step 1: Send debrief to Groq with the exact extraction prompt
    extraction_prompt = (
        'Extract distinct, memory-worthy facts from this meeting debrief. Focus on: '
        'commitments made (by either side), pending follow-ups, concerns or objections '
        'raised, communication style or preferences, and key facts about this contact. '
        'Ignore greetings and filler. Never invent information not present in the text. '
        'Return each fact as a short standalone sentence, one per line, no numbering.\n\n'
        f'Debrief: "{raw_text}"'
    )

    try:
        completion = groq_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional executive assistant extracting distinct, factual meeting debrief memories. Only output memory-worthy facts, one per line, with no bullets or numbers.",
                },
                {"role": "user", "content": extraction_prompt},
            ],
            temperature=0.1,
        )
        output_text = completion.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"Groq extraction failed: {e}")
        raise HTTPException(status_code=502, detail=f"Groq fact extraction failed: {str(e)}")

    # Step 2: Parse into individual facts
    raw_lines = [l.strip() for l in output_text.splitlines() if l.strip()]
    facts = []
    for line in raw_lines:
        cleaned = re.sub(r"^(\d+[\.\)]|[-*•])\s*", "", line).strip()
        if (
            cleaned
            and not cleaned.lower().startswith("no memory-worthy facts")
            and not cleaned.lower().startswith("no facts")
            and not cleaned.lower().startswith("none")
        ):
            facts.append(cleaned)

    # Step 3: Store each extracted fact in the selected contact's Hindsight bank using retain()
    stored_facts = []
    failed_facts = []
    now_utc = datetime.now(timezone.utc)

    for fact in facts:
        try:
            logger.info(f"Retaining in bank '{bank_id}': {fact}")
            hindsight.retain(
                bank_id=bank_id,
                content=fact,
                context="Meeting Debrief",
                timestamp=now_utc,
            )
            stored_facts.append(fact)
        except Exception as e:
            logger.error(f"Hindsight retain error for '{fact}': {e}")
            failed_facts.append({"fact": fact, "error": str(e)})

    # If all fails and there were facts, error out
    if facts and not stored_facts:
        error_detail = failed_facts[0]["error"] if failed_facts else "Unknown Hindsight error"
        raise HTTPException(status_code=502, detail=f"Failed to store memories in Hindsight: {error_detail}")

    return {
        "success": True,
        "bank_id": bank_id,
        "contact_name": contact["name"],
        "extracted_facts": stored_facts,
        "count": len(stored_facts),
        "message": f"Successfully remembered {len(stored_facts)} facts for {contact['name']}.",
    }


@app.get("/api/contacts/{contact_id}/memories")
def get_contact_memories(contact_id: str):
    contact = get_contact_or_404(contact_id)
    bank_id = contact["bank_id"]
    hindsight = get_hindsight_client()

    memories_list = []
    seen_texts = set()

    # 1. Try listing memory units from Hindsight
    try:
        list_res = hindsight.list_memories(bank_id=bank_id, limit=100)
        items = getattr(list_res, "items", []) or []
        for item in items:
            text = getattr(item, "text", "") or ""
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)

            # Date/time
            date_val = None
            for attr in ("mentioned_at", "occurred_start", "var_date", "updated_at"):
                v = getattr(item, attr, None)
                if v:
                    date_val = str(v)
                    break

            # Context/category
            context_val = getattr(item, "context", None) or getattr(item, "fact_type", None) or None

            memories_list.append({
                "id": getattr(item, "id", None),
                "text": text,
                "date": date_val,
                "context": context_val,
            })
    except Exception as e:
        logger.warning(f"list_memories for bank '{bank_id}' returned: {e}")

    # 2. Also perform a broad recall query to ensure any freshly retained memories are retrieved
    try:
        recall_res = hindsight.recall(
            bank_id=bank_id,
            query="all meeting facts commitments follow-ups objections preferences",
            max_tokens=4096,
        )
        results = getattr(recall_res, "results", []) or []
        for res in results:
            text = getattr(res, "text", "") or ""
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)

            date_val = None
            for attr in ("mentioned_at", "occurred_start"):
                v = getattr(res, attr, None)
                if v:
                    date_val = str(v)
                    break

            context_val = getattr(res, "context", None) or getattr(res, "type", None) or None

            memories_list.append({
                "id": getattr(res, "id", None),
                "text": text,
                "date": date_val,
                "context": context_val,
            })
    except Exception as e:
        logger.warning(f"recall query for bank '{bank_id}' returned: {e}")

    return {
        "bank_id": bank_id,
        "contact_name": contact["name"],
        "memories": memories_list,
        "total": len(memories_list),
    }


@app.post("/api/contacts/{contact_id}/proactive")
def get_proactive_insight(contact_id: str):
    contact = get_contact_or_404(contact_id)
    bank_id = contact["bank_id"]
    hindsight = get_hindsight_client()
    groq_client = get_groq_client()
    model = get_groq_model()

    # 1. Automatically perform ONE Hindsight recall
    try:
        recall_res = hindsight.recall(
            bank_id=bank_id,
            query="pending commitments follow-ups concerns preferences upcoming interaction roadmap",
            max_tokens=4096,
        )
        results = getattr(recall_res, "results", []) or []
        recalled_texts = [r.text.strip() for r in results if getattr(r, "text", None) and r.text.strip()]
    except Exception as e:
        logger.error(f"Hindsight recall error for proactive insight: {e}")
        raise HTTPException(status_code=502, detail=f"Hindsight recall failed: {str(e)}")

    # If no memories stored
    if not recalled_texts:
        return {
            "insight": "There isn't enough history yet to provide a useful preparation insight.",
            "memories_count": 0,
        }

    # 2. Perform ONE Groq call with the exact instruction
    formatted_memories = "\n".join(f"- {m}" for m in recalled_texts)
    prompt = (
        "Based on all memories for this contact, what is the ONE most important thing the user should "
        "know or do before their next interaction with them? Focus on pending commitments or follow-ups "
        "if any exist. Keep it to one or two sentences. If there isn't enough memory to say anything useful, "
        "say so plainly.\n\n"
        f"Memories for {contact['name']}:\n"
        f"{formatted_memories}"
    )

    try:
        completion = groq_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an executive preparation advisor. Give a direct, 1-2 sentence proactive insight focused on pending commitments or follow-ups.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        insight_text = completion.choices[0].message.content or ""
        insight_text = insight_text.strip()
    except Exception as e:
        logger.error(f"Groq proactive insight call failed: {e}")
        raise HTTPException(status_code=502, detail=f"Groq preparation insight failed: {str(e)}")

    return {
        "insight": insight_text,
        "memories_count": len(recalled_texts),
    }


@app.post("/api/contacts/{contact_id}/ask")
def ask_question(contact_id: str, payload: AskRequest):
    contact = get_contact_or_404(contact_id)
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    bank_id = contact["bank_id"]
    hindsight = get_hindsight_client()
    groq_client = get_groq_client()
    model = get_groq_model()

    # 1. Call Hindsight recall() exactly once
    try:
        recall_res = hindsight.recall(
            bank_id=bank_id,
            query=question,
            max_tokens=4096,
        )
        results = getattr(recall_res, "results", []) or []
        recalled_texts = [r.text.strip() for r in results if getattr(r, "text", None) and r.text.strip()]
    except Exception as e:
        logger.error(f"Hindsight recall error: {e}")
        raise HTTPException(status_code=502, detail=f"Hindsight recall failed: {str(e)}")

    # If recall returns no relevant memories, the final answer MUST be "Not in memory."
    if not recalled_texts:
        return {
            "answer": "Not in memory.",
            "memories_used": [],
        }

    # 2. Call Groq exactly once with grounded prompt
    formatted_memories = "\n".join(f"- {m}" for m in recalled_texts)
    prompt = (
        "You are answering a question about a contact using ONLY the memories supplied below.\n\n"
        "If the memories do not contain enough information to answer the question, respond EXACTLY:\n\n"
        "Not in memory.\n\n"
        "Never invent facts.\n\n"
        f"Question:\n{question}\n\n"
        f"Memories:\n{formatted_memories}"
    )

    try:
        completion = groq_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a factual assistant. Answer strictly based on the provided memories. If the memories do not contain the answer, respond EXACTLY 'Not in memory.' without elaboration.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        answer = (completion.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error(f"Groq QA call failed: {e}")
        raise HTTPException(status_code=502, detail=f"Groq QA failed: {str(e)}")

    # Check for negative / ungrounded responses
    cleaned_lower = answer.lower().rstrip(".")
    if cleaned_lower in ("not in memory", "not in the memory", "not found in memory"):
        answer = "Not in memory."

    return {
        "answer": answer,
        "memories_used": recalled_texts,
    }


@app.post("/api/contacts/{contact_id}/prepare")
def prepare_next_meeting(contact_id: str):
    contact = get_contact_or_404(contact_id)
    bank_id = contact["bank_id"]
    hindsight = get_hindsight_client()
    groq_client = get_groq_client()
    model = get_groq_model()

    # 1. Retrieve relevant memories from Hindsight
    try:
        recall_res = hindsight.recall(
            bank_id=bank_id,
            query="all previous meetings discussion commitments follow ups objections concerns preferences roadmap pricing",
            max_tokens=4096,
        )
        results = getattr(recall_res, "results", []) or []
        recalled_texts = [r.text.strip() for r in results if getattr(r, "text", None) and r.text.strip()]
    except Exception as e:
        logger.error(f"Hindsight recall error: {e}")
        raise HTTPException(status_code=502, detail=f"Hindsight recall failed: {str(e)}")

    if not recalled_texts:
        return {
            "remembered_facts": [],
            "suggestion": "There isn't enough history yet to support a real suggestion.",
        }

    # 2. Call Groq with the exact prompt
    formatted_memories = "\n".join(f"- {m}" for m in recalled_texts)
    prompt = (
        "Based on what we've learned from previous meetings with this contact, what "
        "should change about how we approach the next meeting? Be specific about what "
        "to follow up on or do differently. Only answer if there is enough memory to "
        "support a real suggestion — if not, say there isn't enough history yet.\n\n"
        f"Memories for {contact['name']}:\n"
        f"{formatted_memories}"
    )

    try:
        completion = groq_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strategic meeting preparation advisor. Provide specific, tactical suggestions grounded directly in the memories.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        suggestion = (completion.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error(f"Groq prepare call failed: {e}")
        raise HTTPException(status_code=502, detail=f"Groq preparation failed: {str(e)}")

    return {
        "remembered_facts": recalled_texts,
        "suggestion": suggestion,
    }


# Frontend delivery
@app.get("/")
def serve_index():
    index_file = Path(__file__).parent / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({"message": "Meeting Prep Agent API running."})

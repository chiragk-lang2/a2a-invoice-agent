import os
import json
import uuid
import hashlib
import sqlite3
import asyncio
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

# Configure these in your hosting provider, never commit secrets to GitHub.
BASE_URL = os.getenv(
    "PUBLIC_BASE_URL", "https://your-service.onrender.com/a2a/"
).rstrip("/") + "/"
A2A_BEARER_TOKEN = os.getenv("A2A_BEARER_TOKEN", "")
AI_API_KEY = os.getenv("AI_API_KEY", "")
AI_MODEL = os.getenv("AI_MODEL", "gemini-2.5-flash")
DATABASE_PATH = os.getenv("DATABASE_PATH", "invoice_agent.db")

BATCH_MEDIA = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSALS_MEDIA = "application/vnd.ga5.invoice-action-proposals+json"
RESULTS_MEDIA = "application/vnd.ga5.invoice-action-results+json"
RECEIPTS_MEDIA = "application/vnd.ga5.invoice-action-receipts+json"
ACTIONS = {
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
}
TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"}
db_lock = asyncio.Lock()


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def digest(value):
    return sha256(canonical(value))


def new_id():
    return str(uuid.uuid4())


def db():
    connection = sqlite3.connect(DATABASE_PATH, timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def initialize_db():
    # The directory containing DATABASE_PATH must be on persistent storage.
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            context_id TEXT NOT NULL,
            principal TEXT NOT NULL,
            batch_id TEXT NOT NULL,
            state TEXT NOT NULL,
            task_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS tasks_owner_idx
            ON tasks(principal, created_at);
        CREATE TABLE IF NOT EXISTS messages (
            principal TEXT NOT NULL,
            message_id TEXT NOT NULL,
            message_hash TEXT NOT NULL,
            task_id TEXT NOT NULL,
            PRIMARY KEY(principal, message_id)
        );
        CREATE TABLE IF NOT EXISTS decisions (
            package_hash TEXT PRIMARY KEY,
            decision_json TEXT NOT NULL
        );
        """)


@asynccontextmanager
async def lifespan(app):
    initialize_db()
    yield


app = FastAPI(title="A2A Invoice Action Agent", version="1.0.0", lifespan=lifespan)


def check_version(request):
    if request.headers.get("A2A-Version") != "1.0":
        raise HTTPException(status_code=400, detail="Unsupported A2A version")


def authenticate(request):
    if not A2A_BEARER_TOKEN:
        raise HTTPException(status_code=503, detail="Server authentication is not configured")
    supplied = request.headers.get("Authorization", "")
    expected = "Bearer " + A2A_BEARER_TOKEN
    if not __import__("hmac").compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Never store or expose the bearer token itself.
    return sha256(A2A_BEARER_TOKEN)


def require_json_media(request):
    media = request.headers.get("Content-Type", "").split(";", 1)[0].strip()
    if media != "application/a2a+json":
        raise HTTPException(status_code=415, detail="Unsupported media type")


def json_response(value, status_code=200):
    encoded = canonical(value).encode("utf-8")
    if len(encoded) > 512 * 1024:
        raise HTTPException(status_code=413, detail="Response too large")
    return JSONResponse(
        content=json.loads(encoded),
        status_code=status_code,
        media_type="application/a2a+json",
    )


def get_parts(message):
    parts = message.get("parts", [])
    return parts if isinstance(parts, list) else []


def part_data(part):
    data = part.get("data")
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return None
    return None


def find_data_part(message, media_type):
    for part in get_parts(message):
        if part.get("mediaType") == media_type:
            return part_data(part)
    return None


def data_part(media_type, data):
    return {"mediaType": media_type, "data": data}


def state_of(task):
    return task.get("status", {}).get("state", "")


def set_state(task, state):
    task["status"] = {"state": state}


def save_task(conn, task, principal, created_at=None):
    stamp = utcnow()
    conn.execute(
        """INSERT INTO tasks(task_id, context_id, principal, batch_id, state,
                            task_json, created_at, updated_at)
           VALUES(?,?,?,?,?,?,?,?)
           ON CONFLICT(task_id) DO UPDATE SET
             state=excluded.state, task_json=excluded.task_json,
             updated_at=excluded.updated_at""",
        (
            task["id"], task["contextId"], principal, task.get("batchId", ""),
            state_of(task), canonical(task),
            created_at or stamp, stamp,
        ),
    )


def load_task(conn, task_id, principal):
    row = conn.execute(
        "SELECT task_json FROM tasks WHERE task_id=? AND principal=?",
        (task_id, principal),
    ).fetchone()
    if row is None:
        # Generic response: do not disclose whether another user's task exists.
        raise HTTPException(status_code=404, detail="Task not found")
    return json.loads(row["task_json"])


def message_hash(message):
    # The configuration is intentionally not included.
    return digest(message)


def idempotent_lookup(conn, principal, message_id, content_hash):
    row = conn.execute(
        "SELECT message_hash, task_id FROM messages WHERE principal=? AND message_id=?",
        (principal, message_id),
    ).fetchone()
    if row is None:
        return None
    if row["message_hash"] != content_hash:
        raise HTTPException(status_code=409, detail="IDEMPOTENCY_CONFLICT")
    return row["task_id"]


def remember_message(conn, principal, message_id, content_hash, task_id):
    conn.execute(
        "INSERT INTO messages(principal,message_id,message_hash,task_id) VALUES(?,?,?,?)",
        (principal, message_id, content_hash, task_id),
    )


def validate_batch(batch):
    if not isinstance(batch, dict):
        raise HTTPException(status_code=400, detail="Invalid batch")
    if not isinstance(batch.get("batchId"), str) or not batch["batchId"]:
        raise HTTPException(status_code=400, detail="Invalid batchId")
    if not isinstance(batch.get("policyRevision"), str) or not batch["policyRevision"]:
        raise HTTPException(status_code=400, detail="Invalid policy revision")
    packages = batch.get("packages")
    if not isinstance(packages, list) or len(packages) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 invoice packages")
    ids = []
    for package in packages:
        if not isinstance(package, dict):
            raise HTTPException(status_code=400, detail="Invalid package")
        package_id = package.get("packageId")
        if not isinstance(package_id, str) or not package_id:
            raise HTTPException(status_code=400, detail="Missing packageId")
        ids.append(package_id)
    if len(ids) != len(set(ids)):
        raise HTTPException(status_code=400, detail="Duplicate package IDs")
    return packages


SYSTEM_PROMPT = """
You are an evidence-based invoice business-action agent. Analyze the supplied
synthetic invoice packages and use ONLY facts and evidence present in them.
Return JSON only, shaped as {"proposals":[...]}.

Choose exactly one action per package:
settle_invoice = valid, reconciled, and within autonomous authority.
request_approval = commercially valid but outside delegated authority.
hold_invoice = payment pauses pending a stated verification.
reject_duplicate = the SAME commercial invoice was already paid.
open_exception = material records conflict and need exception workflow.

A payment for a different invoice is NOT a duplicate. Watch for negation,
old examples, archive examples, irrelevant action words, and training decoys.
For each package cite exactly the three decisive bracketed references from
the paragraph that determines the action. Do not cite cover-sheet references,
archive examples, or training decoys. Do not invent facts or references.
amountMinor is the amount in the smallest currency unit (INR rupees * 100).
If a material fact is unclear or records conflict, do not assume it is true.
Rationale must be 60-1500 characters, name the chosen action, and cite at
least two evidence references.

Each proposal must have:
packageId (exact ID), actionId (unique ID, at least 12 characters),
action (one of the five exact actions), facts {vendorName, invoiceNumber,
amountMinor integer, currency}, evidenceRefs (exactly 3 strings), rationale.
"""


def validate_proposals(batch, proposals):
    packages = batch["packages"]
    if not isinstance(proposals, list) or len(proposals) != len(packages):
        raise ValueError("Expected one proposal per package")
    package_ids = {p["packageId"] for p in packages}
    if {p.get("packageId") for p in proposals} != package_ids:
        raise ValueError("Proposal package IDs do not match")
    seen_actions = set()
    by_package = {}
    for p in proposals:
        if p.get("action") not in ACTIONS:
            raise ValueError("Invalid action")
        aid = p.get("actionId")
        if not isinstance(aid, str) or len(aid) < 12 or aid in seen_actions:
            raise ValueError("Invalid or duplicate actionId")
        seen_actions.add(aid)
        facts = p.get("facts")
        if not isinstance(facts, dict):
            raise ValueError("Invalid facts")
        if not isinstance(facts.get("vendorName"), str) or not facts["vendorName"]:
            raise ValueError("Invalid vendorName")
        if not isinstance(facts.get("invoiceNumber"), str) or not facts["invoiceNumber"]:
            raise ValueError("Invalid invoiceNumber")
        if type(facts.get("amountMinor")) is not int or facts["amountMinor"] < 0:
            raise ValueError("Invalid amountMinor")
        if not isinstance(facts.get("currency"), str) or not facts["currency"]:
            raise ValueError("Invalid currency")
        refs = p.get("evidenceRefs")
        if (not isinstance(refs, list) or len(refs) != 3
                or not all(isinstance(r, str) and r for r in refs)
                or len(set(refs)) != 3):
            raise ValueError("Exactly three distinct evidence references required")
        rationale = p.get("rationale")
        if not isinstance(rationale, str) or not 60 <= len(rationale) <= 1500:
            raise ValueError("Invalid rationale")
        by_package[p["packageId"]] = p
    return [by_package[p["packageId"]] for p in packages]


async def call_gemini(batch):
    if not AI_API_KEY:
        raise HTTPException(status_code=503, detail="AI_API_KEY is not configured")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + AI_MODEL + ":generateContent"
    )
    request_json = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{
            "role": "user",
            "parts": [{
                "text": "Analyze this batch and preserve exact IDs and references:\n"
                        + canonical(batch)
            }],
        }],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }
    async with httpx.AsyncClient(timeout=38) as client:
        response = await client.post(
            url,
            headers={"x-goog-api-key": AI_API_KEY, "Content-Type": "application/json"},
            json=request_json,
        )
        response.raise_for_status()
        result = response.json()
    candidates = result.get("candidates", [])
    if not candidates:
        raise ValueError("Gemini returned no candidates")
    content = "".join(
        part.get("text", "")
        for part in candidates[0].get("content", {}).get("parts", [])
    )
    parsed = json.loads(content)
    return validate_proposals(batch, parsed.get("proposals"))


async def make_proposals(conn, batch):
    packages = batch["packages"]
    cache = {}
    missing = []
    for package in packages:
        key = digest(package)
        row = conn.execute(
            "SELECT decision_json FROM decisions WHERE package_hash=?", (key,)
        ).fetchone()
        if row:
            cache[package["packageId"]] = json.loads(row["decision_json"])
        else:
            missing.append(package)

    # One batched model call for all cache misses.
    if missing:
        partial_batch = dict(batch)
        partial_batch["packages"] = missing
        generated = await call_gemini(partial_batch)
        for package, proposal in zip(missing, generated):
            decision = {
                "action": proposal["action"],
                "facts": proposal["facts"],
                "evidenceRefs": proposal["evidenceRefs"],
                "rationale": proposal["rationale"],
            }
            conn.execute(
                "INSERT OR REPLACE INTO decisions(package_hash,decision_json) VALUES(?,?)",
                (digest(package), canonical(decision)),
            )
            cache[package["packageId"]] = decision

    # IDs are stable for this exact batch/package namespace and unique per package.
    output = []
    for package in packages:
        decision = cache[package["packageId"]]
        action_id = str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            batch["batchId"] + ":" + package["packageId"] + ":" + digest(package),
        ))
        proposal = {
            "packageId": package["packageId"],
            "actionId": action_id,
            **decision,
        }
        output.append(proposal)
    return output


@app.get("/.well-known/agent-card.json")
async def agent_card():
    return {
        "name": "A2A Invoice Action Agent",
        "description": "Evidence-based invoice decisions with approval-gated receipts.",
        "version": "1.0.0",
        "capabilities": {},
        "supportedInterfaces": [{
            "url": BASE_URL,
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": "1.0",
        }],
        "defaultInputModes": [BATCH_MEDIA, RESULTS_MEDIA],
        "defaultOutputModes": [PROPOSALS_MEDIA, RECEIPTS_MEDIA],
        "skills": [{
            "id": "invoice_action_agent",
            "name": "Invoice Action Agent",
            "description": "Analyzes invoice packages and handles approved actions safely.",
            "tags": ["invoice", "accounting", "approval", "evidence"],
        }],
    }


@app.post("/a2a/message:send")
async def send_message(request: Request):
    check_version(request)
    require_json_media(request)
    principal = authenticate(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    if not isinstance(body, dict) or not isinstance(body.get("message"), dict):
        raise HTTPException(status_code=400, detail="Invalid message envelope")
    message = body["message"]
    if message.get("role") != "ROLE_USER":
        raise HTTPException(status_code=400, detail="Invalid role")
    message_id = message.get("messageId")
    if not isinstance(message_id, str) or not message_id:
        raise HTTPException(status_code=400, detail="Missing messageId")
    content_hash = message_hash(message)

    async with db_lock:
        with db() as conn:
            existing_task_id = idempotent_lookup(conn, principal, message_id, content_hash)
            if existing_task_id:
                return json_response({"task": load_task(conn, existing_task_id, principal)})

            batch = find_data_part(message, BATCH_MEDIA)
            if batch is not None:
                validate_batch(batch)
                if message.get("taskId") or message.get("contextId"):
                    raise HTTPException(status_code=400, detail="Unexpected initial task/context ID")
                proposals = await make_proposals(conn, batch)
                task_id, context_id = new_id(), new_id()
                task = {
                    "id": task_id,
                    "contextId": context_id,
                    "status": {"state": "TASK_STATE_INPUT_REQUIRED"},
                    "history": [message],
                    "artifacts": [{
                        "artifactId": new_id(),
                        "name": "invoice-action-proposals",
                        "parts": [data_part(PROPOSALS_MEDIA, {
                            "batchId": batch["batchId"],
                            "proposals": proposals,
                        })],
                    }],
                    "batchId": batch["batchId"],
                }
                save_task(conn, task, principal)
                remember_message(conn, principal, message_id, content_hash, task_id)
                return json_response({"task": task})

            results_data = find_data_part(message, RESULTS_MEDIA)
            if results_data is None:
                raise HTTPException(status_code=400, detail="Unsupported message")
            task_id, context_id = message.get("taskId"), message.get("contextId")
            if not isinstance(task_id, str) or not isinstance(context_id, str):
                raise HTTPException(status_code=400, detail="Missing task/context")
            task = load_task(conn, task_id, principal)
            if task.get("contextId") != context_id:
                raise HTTPException(status_code=409, detail="Context mismatch")
            if state_of(task) in TERMINAL:
                raise HTTPException(status_code=409, detail="Task is terminal")
            if results_data.get("batchId") != task.get("batchId"):
                raise HTTPException(status_code=409, detail="Batch mismatch")

            proposal_data = None
            for artifact in task.get("artifacts", []):
                for part in artifact.get("parts", []):
                    if part.get("mediaType") == PROPOSALS_MEDIA:
                        proposal_data = part.get("data")
            if not isinstance(proposal_data, dict):
                raise HTTPException(status_code=409, detail="Missing proposals")
            proposals = proposal_data.get("proposals", [])
            by_package = {p["packageId"]: p for p in proposals}
            results = results_data.get("results")
            if not isinstance(results, list) or len(results) != len(by_package):
                raise HTTPException(status_code=409, detail="Incomplete result set")

            seen, executions = set(), []
            for result in results:
                if not isinstance(result, dict):
                    raise HTTPException(status_code=400, detail="Invalid result")
                package_id = result.get("packageId")
                if package_id in seen or package_id not in by_package:
                    raise HTTPException(status_code=409, detail="Duplicate or unknown package")
                seen.add(package_id)
                proposal = by_package[package_id]
                if result.get("actionId") != proposal["actionId"]:
                    raise HTTPException(status_code=409, detail="Action ID mismatch")
                if result.get("action") != proposal["action"]:
                    raise HTTPException(status_code=409, detail="Action mismatch")
                if result.get("outcome") not in {"ACCEPTED", "REJECTED"}:
                    raise HTTPException(status_code=400, detail="Invalid outcome")
                if result["outcome"] == "ACCEPTED":
                    nonce = result.get("receiptNonce")
                    if not isinstance(nonce, str) or not nonce:
                        raise HTTPException(status_code=400, detail="Invalid receipt")
                    executions.append({
                        "packageId": package_id,
                        "actionId": proposal["actionId"],
                        "action": proposal["action"],
                        "receiptNonce": nonce,
                        "facts": proposal["facts"],
                        "evidenceRefs": proposal["evidenceRefs"],
                    })

            task["history"].append(message)
            task["artifacts"].append({
                "artifactId": new_id(),
                "name": "invoice-action-receipts",
                "parts": [data_part(RECEIPTS_MEDIA, {
                    "batchId": task["batchId"],
                    "executions": executions,
                })],
            })
            set_state(task, "TASK_STATE_COMPLETED")
            save_task(conn, task, principal)
            remember_message(conn, principal, message_id, content_hash, task_id)
            return json_response({"task": task})


@app.get("/a2a/tasks")
async def list_tasks(request: Request):
    check_version(request)
    principal = authenticate(request)
    with db() as conn:
        rows = conn.execute(
            "SELECT task_json FROM tasks WHERE principal=? ORDER BY created_at DESC",
            (principal,),
        ).fetchall()
    return json_response({"tasks": [json.loads(row["task_json"]) for row in rows]})


@app.get("/a2a/tasks/{task_id}")
async def get_task(task_id: str, request: Request):
    check_version(request)
    principal = authenticate(request)
    with db() as conn:
        task = load_task(conn, task_id, principal)
    return json_response(task)


@app.post("/a2a/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request):
    check_version(request)
    require_json_media(request)
    principal = authenticate(request)
    async with db_lock:
        with db() as conn:
            task = load_task(conn, task_id, principal)
            if state_of(task) in TERMINAL:
                raise HTTPException(status_code=409, detail="Task is terminal")
            set_state(task, "TASK_STATE_CANCELED")
            save_task(conn, task, principal)
            return json_response(task)

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


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = os.getenv(
    "PUBLIC_BASE_URL",
    "https://your-service.onrender.com/a2a/"
).rstrip("/") + "/"

BEARER_TOKEN = os.getenv("A2A_BEARER_TOKEN", "")
AI_API_KEY = os.getenv("AI_API_KEY", "")
AI_BASE_URL = os.getenv(
    "AI_BASE_URL",
    "https://api.openai.com/v1"
).rstrip("/")
AI_MODEL = os.getenv("AI_MODEL", "gpt-4o-mini")

DATABASE = os.getenv("DATABASE_PATH", "invoice_agent.db")

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

TERMINAL_STATES = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"}

db_lock = asyncio.Lock()


# ============================================================
# DATABASE
# ============================================================

def connect_db():
    conn = sqlite3.connect(DATABASE, timeout=20)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db():
    with connect_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                context_id TEXT NOT NULL,
                principal TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                state TEXT NOT NULL,
                task_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                principal TEXT NOT NULL,
                message_id TEXT NOT NULL,
                message_hash TEXT NOT NULL,
                task_id TEXT NOT NULL,
                PRIMARY KEY (principal, message_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                package_hash TEXT PRIMARY KEY,
                proposal_json TEXT NOT NULL
            )
        """)

        conn.commit()


@asynccontextmanager
async def lifespan(app):
    initialize_db()
    yield


app = FastAPI(
    title="A2A Invoice Action Agent",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# HELPERS
# ============================================================

def now():
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def digest(value):
    return hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


def new_id():
    return str(uuid.uuid4())


def extract_parts(message):
    return message.get("parts", [])


def extract_data(part):
    # A2A data parts can be represented as {"data": {...}}.
    # Accept JSON text too, when supplied by a compatible client.
    data = part.get("data")

    if isinstance(data, dict):
        return data

    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return None

    return None


def find_part(message, media_type):
    for part in extract_parts(message):
        if part.get("mediaType") == media_type:
            data = extract_data(part)
            if data is not None:
                return data

    return None


def make_text_part(text):
    return {
        "text": text,
        "mediaType": "text/plain",
    }


def make_data_part(media_type, data):
    return {
        "data": data,
        "mediaType": media_type,
    }


def make_task_message(message_id, role, parts):
    return {
        "messageId": message_id,
        "role": role,
        "parts": parts,
    }


def get_state(task):
    return task.get("status", {}).get("state", "")


def set_state(task, state):
    task["status"] = {"state": state}


def validate_proposals(batch, proposals):
    packages = batch.get("packages", [])

    if not isinstance(packages, list) or len(packages) != 12:
        raise ValueError("Expected exactly 12 invoice packages")

    package_ids = []
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError("Invalid package")
        pid = package.get("packageId")
        if not isinstance(pid, str) or not pid:
            raise ValueError("Missing packageId")
        package_ids.append(pid)

    if len(set(package_ids)) != len(package_ids):
        raise ValueError("Duplicate package IDs")

    if not isinstance(proposals, list) or len(proposals) != len(packages):
        raise ValueError("One proposal is required per package")

    if {p.get("packageId") for p in proposals} != set(package_ids):
        raise ValueError("Proposal package IDs do not match the batch")

    action_ids = set()

    for proposal in proposals:
        if proposal.get("action") not in ACTIONS:
            raise ValueError("Invalid action")

        action_id = proposal.get("actionId")
        if not isinstance(action_id, str) or len(action_id) < 12:
            raise ValueError("Invalid actionId")

        if action_id in action_ids:
            raise ValueError("Duplicate actionId")
        action_ids.add(action_id)

        facts = proposal.get("facts")
        if not isinstance(facts, dict):
            raise ValueError("Invalid facts")

        if not isinstance(facts.get("vendorName"), str):
            raise ValueError("Invalid vendorName")
        if not isinstance(facts.get("invoiceNumber"), str):
            raise ValueError("Invalid invoiceNumber")
        if type(facts.get("amountMinor")) is not int:
            raise ValueError("Invalid amountMinor")
        if not isinstance(facts.get("currency"), str):
            raise ValueError("Invalid currency")

        refs = proposal.get("evidenceRefs")
        if (
            not isinstance(refs, list)
            or len(refs) != 3
            or not all(isinstance(r, str) and r for r in refs)
            or len(set(refs)) != 3
        ):
            raise ValueError("Exactly three unique evidence references required")

        rationale = proposal.get("rationale")
        if not isinstance(rationale, str) or not 60 <= len(rationale) <= 1500:
            raise ValueError("Invalid rationale length")

        if proposal.get("batchId") not in (None, batch.get("batchId")):
            raise ValueError("Wrong proposal batch")

    return proposals


# ============================================================
# AI DECISION LAYER
# ============================================================

SYSTEM_PROMPT = """
You are an evidence-based business invoice decision agent.

Read the supplied invoice packages and their documents carefully.

For every package, return exactly one proposal.

Allowed actions:
- settle_invoice: valid, reconciled and within autonomous authority
- request_approval: commercially valid but outside delegated authority
- hold_invoice: payment must pause until verification
- reject_duplicate: the SAME commercial invoice has already been paid
- open_exception: material records conflict and require escalation

Important:
- A payment for a different invoice is NOT a duplicate.
- Do not confuse historical examples, negated statements or irrelevant
  action words with current decisive evidence.
- Use the exact three decisive bracketed evidence references from the
  paragraph that determines the action.
- Do not include cover-sheet references, archive examples or training decoys.
- Never invent facts, invoice numbers, amounts, currencies or references.
- amountMinor is the amount in the smallest currency unit.
  For INR, 100 rupees is 10000 paise.
- Give a rationale between 60 and 1500 characters.
- Name the selected action in the rationale and cite at least two references.
- Generate a unique actionId of at least 12 characters for every proposal.
- Use only evidence present in the supplied package.
- If a material fact is unclear or contradictory, do not assume it is true.

Return ONLY a JSON object of this form:
{
  "proposals": [
    {
      "packageId": "exact package ID",
      "actionId": "unique durable identifier",
      "action": "one allowed action",
      "facts": {
        "vendorName": "vendor",
        "invoiceNumber": "invoice number",
        "amountMinor": 12345,
        "currency": "INR"
      },
      "evidenceRefs": ["reference 1", "reference 2", "reference 3"],
      "rationale": "reason for the decision"
    }
  ]
}
"""


async def call_ai(batch):
    if not AI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="AI_API_KEY is not configured",
        )

    url = AI_BASE_URL + "/chat/completions"

    payload = {
        "model": AI_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Analyze this invoice batch. Preserve exact package "
                    "IDs and evidence references.\n\n"
                    + canonical_json(batch)
                ),
            },
        ],
        "response_format": {"type": "json_object"},
    }

    try:
        async with httpx.AsyncClient(timeout=35) as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {AI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

        content = result["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        proposals = parsed["proposals"]

        return validate_proposals(batch, proposals)

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Invoice analysis failed",
        )


async def decide_batch(batch):
    packages = batch.get("packages")

    if not isinstance(packages, list) or len(packages) != 12:
        raise HTTPException(
            status_code=400,
            detail="Invalid invoice batch",
        )

    proposals = []

    # Cache decisions by canonical package content, not batch/task ID.
    with connect_db() as conn:
        for package in packages:
            package_hash = digest(package)
            row = conn.execute(
                "SELECT proposal_json FROM decisions WHERE package_hash = ?",
                (package_hash,),
            ).fetchone()

            if row:
                proposal = json.loads(row["proposal_json"])
                proposals.append(proposal)
            else:
                # A missing decision is handled by one batched AI call below.
                proposals = []
                break

    if proposals:
        # Cached decisions may have originated from another batch.
        # Refresh the batch ID and action IDs for this task.
        for proposal in proposals:
            proposal["actionId"] = new_id()
        return proposals

    proposals = await call_ai(batch)

    with connect_db() as conn:
        for package, proposal in zip(packages, proposals):
            conn.execute(
                """
                INSERT OR IGNORE INTO decisions(package_hash, proposal_json)
                VALUES (?, ?)
                """,
                (digest(package), canonical_json(proposal)),
            )
        conn.commit()

    return proposals


# ============================================================
# TASK PERSISTENCE
# ============================================================

def save_task(task, principal):
    timestamp = now()

    with connect_db() as conn:
        conn.execute(
            """
            INSERT INTO tasks(
                task_id, context_id, principal, batch_id,
                state, task_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                state=excluded.state,
                task_json=excluded.task_json,
                updated_at=excluded.updated_at
            """,
            (
                task["id"],
                task["contextId"],
                principal,
                task.get("batchId", ""),
                get_state(task),
                canonical_json(task),
                timestamp,
                timestamp,
            ),
        )
        conn.commit()


def load_task(task_id, principal):
    with connect_db() as conn:
        row = conn.execute(
            """
            SELECT task_json FROM tasks
            WHERE task_id = ? AND principal = ?
            """,
            (task_id, principal),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Task not found")

    return json.loads(row["task_json"])


def authenticated_principal(request):
    if not BEARER_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Server authentication is not configured",
        )

    authorization = request.headers.get("Authorization", "")
    expected = "Bearer " + BEARER_TOKEN

    if not hashlib.sha256(authorization.encode()).digest() == \
       hashlib.sha256(expected.encode()).digest():
        raise HTTPException(status_code=401, detail="Unauthorized")

    return hashlib.sha256(BEARER_TOKEN.encode()).hexdigest()


def check_protocol(request):
    if request.headers.get("A2A-Version") != "1.0":
        raise HTTPException(status_code=400, detail="Unsupported version")

    content_type = request.headers.get("Content-Type", "")
    if content_type.split(";")[0].strip() != "application/a2a+json":
        raise HTTPException(status_code=415, detail="Unsupported media type")


def response_json(data, status=200):
    body = canonical_json(data).encode("utf-8")

    if len(body) > 512 * 1024:
        raise HTTPException(status_code=413, detail="Response too large")

    return JSONResponse(
        content=json.loads(body),
        status_code=status,
        media_type="application/a2a+json",
    )


# ============================================================
# AGENT CARD
# ============================================================

@app.get("/.well-known/agent-card.json")
async def agent_card():
    return {
        "name": "A2A Invoice Action Agent",
        "description": (
            "Analyzes invoice packages and proposes evidence-based "
            "business actions with safe approval and receipt handling."
        ),
        "version": "1.0.0",
        "capabilities": {},
        "supportedInterfaces": [
            {
                "url": BASE_URL,
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0",
            }
        ],
        "defaultInputModes": [
            BATCH_MEDIA,
            RESULTS_MEDIA,
        ],
        "defaultOutputModes": [
            PROPOSALS_MEDIA,
            RECEIPTS_MEDIA,
        ],
        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "Invoice Action Agent",
                "description": (
                    "Reviews invoices, cites evidence, and executes "
                    "only accepted proposals."
                ),
                "tags": [
                    "invoice",
                    "accounting",
                    "approval",
                    "evidence",
                ],
            }
        ],
    }


# ============================================================
# INITIAL MESSAGE AND RESULT CONTINUATION
# ============================================================

@app.post("/a2a/message:send")
async def send_message(request: Request):
    check_protocol(request)
    principal = authenticated_principal(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    message = body.get("message")
    if not isinstance(message, dict):
        raise HTTPException(status_code=400, detail="Invalid message")

    if message.get("role") != "ROLE_USER":
        raise HTTPException(status_code=400, detail="Invalid role")

    message_id = message.get("messageId")
    if not isinstance(message_id, str) or not message_id:
        raise HTTPException(status_code=400, detail="Missing messageId")

    message_hash = digest(message)

    # Message idempotency: same principal + message ID + same message.
    with connect_db() as conn:
        existing = conn.execute(
            """
            SELECT message_hash, task_id FROM messages
            WHERE principal = ? AND message_id = ?
            """,
            (principal, message_id),
        ).fetchone()

    if existing:
        if existing["message_hash"] != message_hash:
            raise HTTPException(
                status_code=409,
                detail="IDEMPOTENCY_CONFLICT",
            )

        task = load_task(existing["task_id"], principal)
        return response_json({"task": task})

    # Initial batch request.
    batch = find_part(message, BATCH_MEDIA)

    if batch is not None:
        if message.get("taskId") or message.get("contextId"):
            raise HTTPException(
                status_code=400,
                detail="Initial message must not specify task IDs",
            )

        batch_id = batch.get("batchId")
        policy_revision = batch.get("policyRevision")

        if not isinstance(batch_id, str) or not batch_id:
            raise HTTPException(status_code=400, detail="Invalid batchId")

        if not isinstance(policy_revision, str) or not policy_revision:
            raise HTTPException(
                status_code=400,
                detail="Invalid policy revision",
            )

        # Ensure no concurrent initial request with this message ID
        # can create a second task while the AI call is running.
        async with db_lock:
            with connect_db() as conn:
                existing = conn.execute(
                    """
                    SELECT message_hash, task_id FROM messages
                    WHERE principal = ? AND message_id = ?
                    """,
                    (principal, message_id),
                ).fetchone()

                if existing:
                    if existing["message_hash"] != message_hash:
                        raise HTTPException(
                            status_code=409,
                            detail="IDEMPOTENCY_CONFLICT",
                        )
                    return response_json({
                        "task": load_task(existing["task_id"], principal)
                    })

                proposals = await decide_batch(batch)

                task_id = new_id()
                context_id = message.get("contextId") or new_id()

                task = {
                    "id": task_id,
                    "contextId": context_id,
                    "status": {"state": "TASK_STATE_INPUT_REQUIRED"},
                    "history": [message],
                    "artifacts": [
                        {
                            "artifactId": new_id(),
                            "name": "invoice-action-proposals",
                            "parts": [
                                make_data_part(
                                    PROPOSALS_MEDIA,
                                    {
                                        "batchId": batch_id,
                                        "proposals": proposals,
                                    },
                                )
                            ],
                        }
                    ],
                    "batchId": batch_id,
                }

                # Save the task and idempotency record before responding.
                with connect_db() as conn:
                    conn.execute(
                        """
                        INSERT INTO tasks(
                            task_id, context_id, principal, batch_id,
                            state, task_json, created_at, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            task_id,
                            context_id,
                            principal,
                            batch_id,
                            get_state(task),
                            canonical_json(task),
                            now(),
                            now(),
                        ),
                    )

                    conn.execute(
                        """
                        INSERT INTO messages(
                            principal, message_id, message_hash, task_id
                        )
                        VALUES (?, ?, ?, ?)
                        """,
                        (principal, message_id, message_hash, task_id),
                    )
                    conn.commit()

        return response_json({"task": task})

    # Result continuation.
    results_data = find_part(message, RESULTS_MEDIA)

    if results_data is None:
        raise HTTPException(status_code=400, detail="Unsupported message")

    task_id = message.get("taskId")
    context_id = message.get("contextId")

    if not isinstance(task_id, str) or not isinstance(context_id, str):
        raise HTTPException(status_code=400, detail="Missing task/context")

    async with db_lock:
        task = load_task(task_id, principal)

        if task.get("contextId") != context_id:
            raise HTTPException(status_code=409, detail="Context mismatch")

        if get_state(task) in TERMINAL_STATES:
            # Exact message replays are handled by the idempotency table.
            raise HTTPException(status_code=409, detail="Task is terminal")

        if results_data.get("batchId") != task.get("batchId"):
            raise HTTPException(status_code=409, detail="Batch mismatch")

        proposal_artifact = next(
            (
                a for a in task.get("artifacts", [])
                if any(
                    p.get("mediaType") == PROPOSALS_MEDIA
                    for p in a.get("parts", [])
                )
            ),
            None,
        )

        if not proposal_artifact:
            raise HTTPException(status_code=409, detail="Missing proposals")

        proposals_data = next(
            p["data"]
            for p in proposal_artifact["parts"]
            if p.get("mediaType") == PROPOSALS_MEDIA
        )

        proposals_by_id = {
            p["packageId"]: p
            for p in proposals_data["proposals"]
        }

        results = results_data.get("results")
        if not isinstance(results, list):
            raise HTTPException(status_code=400, detail="Invalid results")

        seen = set()
        executions = []

        for result in results:
            if not isinstance(result, dict):
                raise HTTPException(status_code=400, detail="Invalid result")

            package_id = result.get("packageId")
            if package_id in seen:
                raise HTTPException(status_code=409, detail="Duplicate result")
            seen.add(package_id)

            proposal = proposals_by_id.get(package_id)
            if not proposal:
                raise HTTPException(status_code=409, detail="Unknown package")

            if result.get("actionId") != proposal["actionId"]:
                raise HTTPException(status_code=409, detail="Action ID mismatch")

            if result.get("action") != proposal["action"]:
                raise HTTPException(status_code=409, detail="Action mismatch")

            if result.get("outcome") not in {"ACCEPTED", "REJECTED"}:
                raise HTTPException(status_code=400, detail="Invalid outcome")

            if result.get("outcome") == "ACCEPTED":
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

        receipts = {
            "batchId": task["batchId"],
            "executions": executions,
        }

        task["artifacts"].append({
            "artifactId": new_id(),
            "name": "invoice-action-receipts",
            "parts": [
                make_data_part(RECEIPTS_MEDIA, receipts)
            ],
        })

        set_state(task, "TASK_STATE_COMPLETED")
        save_task(task, principal)

        with connect_db() as conn:
            conn.execute(
                """
                INSERT INTO messages(
                    principal, message_id, message_hash, task_id
                )
                VALUES (?, ?, ?, ?)
                """,
                (principal, message_id, message_hash, task_id),
            )
            conn.commit()

        return response_json({"task": task})


# ============================================================
# TASK READS
# ============================================================

@app.get("/a2a/tasks")
async def list_tasks(request: Request):
    principal = authenticated_principal(request)

    with connect_db() as conn:
        rows = conn.execute(
            """
            SELECT task_json FROM tasks
            WHERE principal = ?
            ORDER BY created_at DESC
            """,
            (principal,),
        ).fetchall()

    return response_json({
        "tasks": [json.loads(row["task_json"]) for row in rows]
    })


@app.get("/a2a/tasks/{task_id}")
async def get_task(task_id: str, request: Request):
    principal = authenticated_principal(request)
    return response_json(load_task(task_id, principal))


# ============================================================
# CANCELLATION
# ============================================================

@app.post("/a2a/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request):
    check_protocol(request)
    principal = authenticated_principal(request)

    async with db_lock:
        task = load_task(task_id, principal)

        if get_state(task) in TERMINAL_STATES:
            raise HTTPException(status_code=409, detail="Task is terminal")

        set_state(task, "TASK_STATE_CANCELED")
        save_task(task, principal)

        return response_json(task)

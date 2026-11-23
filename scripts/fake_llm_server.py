"""A stand-in for the vLLM endpoint, for local development and end-to-end tests.

Answers the prompts in llm/evals.json with their ideal replies and anything
else with an "unsupported" reply, using the same OpenAI-compatible API as
vLLM. Lets the frontend and /query be built before a GPU endpoint exists.

    uvicorn scripts.fake_llm_server:app --port 8800
    export LLM_BASE_URL=http://localhost:8800/v1 LLM_MODEL=fake
"""
import json
import sys
from pathlib import Path

from fastapi import FastAPI, Request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.evals import load_cases  # noqa: E402

IDEAL = {c["prompt"]: c["ideal"] for c in load_cases()}
FALLBACK = {
    "intent": "unsupported", "explanation": "The stand-in model only knows the eval prompts.",
    "assumptions": [], "horizon_months": None, "scenarios": [],
    "unsupported_reason": "Stand-in model: try a prompt from llm/evals.json.",
}
app = FastAPI(title="Stand-in LLM")


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    prompt = next((m["content"] for m in body["messages"] if m["role"] == "user"), "")
    content = json.dumps(IDEAL.get(prompt, FALLBACK))
    return {
        "id": "fake", "object": "chat.completion", "model": body.get("model", "fake"),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": len(json.dumps(body)) // 4, "completion_tokens": len(content) // 4},
    }

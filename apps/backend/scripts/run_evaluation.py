"""
Run the evaluation benchmark against a live backend + a signed-in user's
documents, and print an aggregate report.

Usage:
    python scripts/run_evaluation.py --email you@example.com --password ***

Assumes:
  - The API is reachable (default http://localhost:8000, override with --api-url)
  - The user already has documents uploaded (see docs/EVALUATION.md for how
    to seed the demo corpus)

This hits the real HTTP API end-to-end (auth -> chat session -> ask), rather
than importing use cases directly, so it also exercises the API layer and
SSE streaming path -- closer to what a real user experiences.
"""
from __future__ import annotations

import argparse
import asyncio
import json

import httpx


async def run(api_url: str, email: str, password: str, collection_id: str | None) -> None:
    async with httpx.AsyncClient(base_url=api_url, timeout=60.0) as client:
        login = await client.post("/auth/login", json={"email": email, "password": password})
        login.raise_for_status()
        token = login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        docs_resp = await client.get("/documents", headers=headers)
        docs_resp.raise_for_status()
        documents = docs_resp.json()
        print(f"Found {len(documents)} documents for this user.")

        session_resp = await client.post(
            "/chat/sessions", json={"collection_id": collection_id, "title": "Evaluation run"}, headers=headers
        )
        session_resp.raise_for_status()
        session_id = session_resp.json()["id"]

        queries = [
            "How many paid sick leave days do employees get per year?",
            "What is the process for requesting reimbursement for a team lunch?",
            "What documents do I have available?",
        ]

        results = []
        for query in queries:
            print(f"\n--- {query}")
            answer_text = ""
            confidence = None
            citation_count = 0
            async with client.stream(
                "POST", "/chat/ask",
                json={"session_id": session_id, "collection_id": collection_id, "question": query},
                headers=headers,
            ) as response:
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[len("data: "):])
                    if event["type"] == "token":
                        answer_text += event["data"]
                    elif event["type"] == "status":
                        print(f"  [{event['data']}]")
                    elif event["type"] == "final":
                        confidence = event["data"]["confidence"]
                        citation_count = len(event["data"]["citations"])
            print(f"  Answer: {answer_text[:200]}{'...' if len(answer_text) > 200 else ''}")
            print(f"  Confidence: {confidence}  Citations: {citation_count}")
            results.append({"query": query, "confidence": confidence, "citations": citation_count})

        print("\n=== Summary ===")
        avg_conf = sum(r["confidence"] or 0 for r in results) / len(results)
        print(f"Queries run: {len(results)}")
        print(f"Average confidence: {avg_conf:.2f}")
        print(f"Queries with >=1 citation: {sum(1 for r in results if r['citations'] > 0)}/{len(results)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation benchmark")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--collection-id", default=None)
    args = parser.parse_args()

    asyncio.run(run(args.api_url, args.email, args.password, args.collection_id))


if __name__ == "__main__":
    main()

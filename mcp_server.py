
import hmac
import json
import os

from mcp.server.fastmcp import FastMCP
from starlette.responses import JSONResponse

from rag_service import get_relevant_context

mcp = FastMCP(
    "College RAG Assistant",
    instructions=(
        "Search college documents to answer questions. "
        "Use retrieved passages as evidence, cite filenames and page "
        "numbers, and never invent information."
    ),
)


@mcp.tool()
def search_college_documents(
    question: str,
    top_k: int = 4,
) -> str:
    """Search indexed college PDFs and return relevant passages with citations."""
    if not question.strip():
        return json.dumps({
            "answer": "Please provide a question.",
            "sources": [],
        })

    top_k = max(1, min(int(top_k), 8))
    result = get_relevant_context(question, top_k=top_k)

    if not result["sources"]:
        return json.dumps({
            "answer": "Information not found in the available documents.",
            "sources": [],
        }, ensure_ascii=False)

    return json.dumps({
        "instructions": (
            "Answer using only the supplied passages. If they do not "
            "provide enough evidence, say the information was not found. "
            "Cite the filename and page number."
        ),
        "sources": result["sources"],
    }, ensure_ascii=False)


if __name__ == "__main__":
    import uvicorn

    expected_token = os.getenv("MCP_AUTH_TOKEN")
    if not expected_token:
        raise RuntimeError(
            "MCP_AUTH_TOKEN must be configured before starting the server."
        )

    app = mcp.streamable_http_app()

    @app.middleware("http")
    async def require_bearer_token(request, call_next):
        authorization = request.headers.get("authorization", "")
        scheme, separator, supplied_token = authorization.partition(" ")

        if (
            scheme.lower() != "bearer"
            or not separator
            or not hmac.compare_digest(supplied_token, expected_token)
        ):
            return JSONResponse(
                {"error": "Authentication required"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await call_next(request)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
    )

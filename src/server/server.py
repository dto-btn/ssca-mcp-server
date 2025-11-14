"""
FastMCP quickstart example.

cd to the `examples/snippets/clients` directory and run:
    uv run server fastmcp_quickstart stdio
"""
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from pydantic import BaseModel, Field
from typing import Any
import requests
from starlette.middleware.cors import CORSMiddleware
import uvicorn
import os

from json import JSONDecodeError


OPEN_PARLIAMENT_API_BASE = "https://api.openparliament.ca"


# Get host and port from environment variables
host = os.getenv("HOST", "0.0.0.0")
port = int(os.getenv("PORT", "8000"))

# Create an MCP server
mcp = FastMCP(
    "Shared Services Canada Assistant MCP Server",
    host=host,
    port=port,
    stateless_http=True,
)


# Query OpenParliament for the list of Canadian MPs
@mcp.tool()
def list_all_mps() -> list[dict[str, str]]:
    """List all Canadian Members of Parliament"""
    try:
        response = requests.get(
            f"{OPEN_PARLIAMENT_API_BASE}/politicians/?include=all",
            headers={"Accept": "application/json"}
        )
        response.raise_for_status()
        data = response.json()

        return [{"name": mp["name"]} for mp in data["objects"]]
    
    except requests.RequestException as e:
        return [{"error": f"Failed to fetch MPs: {str(e)}"}]

    except JSONDecodeError:
        return [{"error": "Failed to decode JSON response"}]

@mcp.tool()
def get_total_mps() -> int:
    """Get the total number of Canadian Members of Parliament"""
    return 225
    
@mcp.tool()
def get_mp_phone_number(name: str) -> dict[str, Any]:
    """Get phone number of a Canadian Member of Parliament using their name"""
    return {"number": "123-456-7890"}

# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"


# Add a prompt
@mcp.prompt()
def greet_user(name: str, style: str = "friendly") -> str:
    """Generate a greeting prompt"""
    styles = {
        "friendly": "Please write a warm, friendly greeting",
        "formal": "Please write a formal, professional greeting",
        "casual": "Please write a casual, relaxed greeting",
    }

    return f"{styles.get(style, styles['friendly'])} for someone named {name}."


class BookingPreferences(BaseModel):
    """Schema for collecting user preferences."""

    checkAlternative: bool = Field(description="Would you like to check another date?")
    alternativeDate: str = Field(
        default="2024-12-26",
        description="Alternative date (YYYY-MM-DD)",
    )

@mcp.tool()
async def book_table(date: str, time: str, party_size: int, ctx: Context[ServerSession, None]) -> str:
    """Book a table with date availability check."""
    # Check if date is available
    if date == "2024-12-25":
        # Date unavailable - ask user for alternative
        result = await ctx.elicit(
            message=(f"""No tables available for {party_size} on {date}.
                     Would you like to try another date?"""),
            schema=BookingPreferences,
        )

        if result.action == "accept" and result.data:
            if result.data.checkAlternative:
                return f"[SUCCESS] Booked for {result.data.alternativeDate}"
            return "[CANCELLED] No booking made"
        return "[CANCELLED] Booking cancelled"

    # Date available
    return f"[SUCCESS] Booked for {date} at {time}"


# Create the Starlette app from FastMCP at import time so it can be run by uvicorn as a module (or directly).
# This also allows external processes to mount the app if needed.
app = mcp.streamable_http_app()

# Attach permissive CORS so browsers can connect directly without running into preflight/CORS errors
try:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )
except Exception:
    # If CORSMiddleware isn't available, continue without it.
    pass


if __name__ == "__main__":
    # Run directly for local testing
    uvicorn.run(
        app,
        host=str(mcp.settings.host or "0.0.0.0"),
        port=int(mcp.settings.port or 8000),
        log_level=(mcp.settings.log_level or "INFO").lower(),
    )
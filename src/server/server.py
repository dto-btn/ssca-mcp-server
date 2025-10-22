from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
import requests
from json import JSONDecodeError
from pydantic import BaseModel, Field

OPEN_PARLIAMENT_API_BASE = "https://api.openparliament.ca"

# Stateful server (maintains session state)
mcp = FastMCP("StatefulServer")

# Other configuration options:
# Stateless server (no session persistence)
# mcp = FastMCP("StatelessServer", stateless_http=True)

# Stateless server (no session persistence, no sse stream with supported client)
# mcp = FastMCP("StatelessServer", stateless_http=True, json_response=True)


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

# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

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


# Run server with streamable_http transport
if __name__ == "__main__":
    mcp.run(transport="streamable-http")

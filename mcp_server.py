"""
MCP Server for triggering bash scripts.

This server provides a tool to execute bash commands from the command line.
"""

import subprocess
from mcp.server.fastmcp import FastMCP
from datetime import datetime
import os

# Create the MCP server instance
mcp = FastMCP("Bash Script Runner", json_response=True)

@mcp.tool()
def run_bash_command(command: str) -> str:
    """
    Execute a bash command and return the output.

    Args:
        command: The bash command to execute

    Returns:
        The stdout output of the command, or stderr if stdout is empty
    """
    try:
        # Run the command with shell=True to allow bash syntax
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )

        # Return stdout if available, otherwise stderr
        output = result.stdout.strip() if result.stdout else result.stderr.strip()

        # Include exit code if non-zero
        if result.returncode != 0:
            output = f"Command failed with exit code {result.returncode}:\n{output}"

        return output

    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds"
    except Exception as e:
        return f"Error executing command: {str(e)}"

@mcp.tool()
def run_daily_process():
    """Runs the daily batch process and creates a DLY_DDMMYYYY_HHMMSS.csv file"""
    result = subprocess.run(
        ["/workspaces/codespaces-demo/batch/scripts/daily_process.bat"],
        capture_output=True,
        text=True,
        timeout=30,
        shell=True
    )
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.returncode
    }

@mcp.tool()
def run_monthly_process():
    """Runs the monthly batch process and creates a MTH_DDMMYYYY_HHMMSS.csv file"""
    result = subprocess.run(
        ["/workspaces/codespaces-demo/batch/scripts/monthly_process.bat"],
        capture_output=True,
        text=True,
        timeout=30,
        shell=True
    )
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.returncode
    }

if __name__ == "__main__":
    # Run the server
    mcp.run()
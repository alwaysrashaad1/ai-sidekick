# sidekick_tools.py
# This module defines all the tools that the Sidekick AI can use, 
# including web browsing, Python execution, file management, search, and push notifications.

from playwright.async_api import async_playwright  # Async API for controlling browsers
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit  # LangChain toolkit for browser automation
from dotenv import load_dotenv  # Loads environment variables from a .env file
import os
import requests  # For sending HTTP requests (used for push notifications)
from langchain.agents import Tool  # Base class for creating tools in LangChain
from langchain_community.agent_toolkits import FileManagementToolkit  # File management toolkit
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun  # Wikipedia query tool
from langchain_experimental.tools import PythonREPLTool  # Allows executing Python code
from langchain_community.utilities import GoogleSerperAPIWrapper  # Wrapper for Google search
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper  # Wikipedia API wrapper

# Load environment variables from .env file
load_dotenv(override=True)

# Push notification credentials from environment variables
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"

# Initialize Google search API wrapper
serper = GoogleSerperAPIWrapper()


# Playwright Tools
# Asynchronously start a Playwright browser instance and return the tools from the toolkit
async def playwright_tools():
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    return toolkit.get_tools(), browser, playwright


# Push Notification Tool
def push(text: str):
    """Send a push notification to the user via Pushover"""
    requests.post(
        pushover_url,
        data={
            "token": pushover_token,
            "user": pushover_user,
            "message": text
        }
    )
    return "success"


# File Management Tools
def get_file_tools():
    """Create sandboxed file management tools for the AI to safely read/write files"""
    toolkit = FileManagementToolkit(root_dir="sandbox")  # Restrict operations to sandbox folder
    return toolkit.get_tools()


# Other Tools Collection
async def other_tools():
    """Assemble additional tools for Sidekick AI"""
    
    # Push notification tool
    push_tool = Tool(
        name="send_push_notification",
        func=push,
        description="Use this tool when you want to send a push notification"
    )
    
    # File management tools
    file_tools = get_file_tools()

    # Web search tool using Google Serper
    tool_search = Tool(
        name="search",
        func=serper.run,
        description="Use this tool when you want to get the results of an online web search"
    )

    # Wikipedia query tool
    wikipedia = WikipediaAPIWrapper()
    wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia)

    # Python REPL tool to execute code dynamically
    python_repl = PythonREPLTool()
    
    # Combine all tools into a single list
    return file_tools + [push_tool, tool_search, python_repl, wiki_tool]

# sidekick.py
# This module defines the Sidekick AI, which acts as an autonomous assistant
# capable of using tools, evaluating its work, and interacting with users
# via a workflow graph. It integrates LangChain tools, a worker LLM, and an
# evaluator LLM with structured outputs.

from typing import Annotated, List, Any, Optional, Dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sidekick_tools import playwright_tools, other_tools  # Custom tools module
import uuid
import asyncio
from datetime import datetime

# Load environment variables from .env
load_dotenv(override=True)


# Define State for Graph
class State(TypedDict):
    """Represents the state object passed between nodes in the workflow graph"""
    messages: Annotated[List[Any], add_messages]  # Chat history, includes system, AI, and human messages
    success_criteria: str  # The goal criteria for the task
    feedback_on_work: Optional[str]  # Feedback from evaluator if previous attempt failed
    success_criteria_met: bool  # Whether the task has been successfully completed
    user_input_needed: bool  # Whether more input from user is required



# Define Structured Output for Evaluator
class EvaluatorOutput(BaseModel):
    """Defines the structured output format for the evaluator LLM"""
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(
        description="True if more input is needed from the user, or clarifications, or the assistant is stuck"
    )


# Sidekick Class
class Sidekick:
    """Encapsulates the AI assistant logic, including tools, LLMs, workflow graph, and memory"""
    def __init__(self):
        # Worker LLM bound to tools
        self.worker_llm_with_tools = None
        # Evaluator LLM with structured output
        self.evaluator_llm_with_output = None
        # Tools available to the AI
        self.tools = None
        self.llm_with_tools = None
        # Workflow graph
        self.graph = None
        # Unique ID for this Sidekick instance
        self.sidekick_id = str(uuid.uuid4())
        # Memory saver to persist graph state/checkpoints
        self.memory = MemorySaver()
        # Browser objects for Playwright tools
        self.browser = None
        self.playwright = None

    # Setup Sidekick
    async def setup(self):
        """
        Initialize the tools, worker and evaluator LLMs, and build the workflow graph.
        Must be awaited because Playwright tools are asynchronous.
        """
        # Load browser and other tools
        self.tools, self.browser, self.playwright = await playwright_tools()
        self.tools += await other_tools()
        
        # Initialize worker LLM and bind tools
        worker_llm = ChatOpenAI(model="gpt-4o-mini")
        self.worker_llm_with_tools = worker_llm.bind_tools(self.tools)
        
        # Initialize evaluator LLM with structured output
        evaluator_llm = ChatOpenAI(model="gpt-4o-mini")
        self.evaluator_llm_with_output = evaluator_llm.with_structured_output(EvaluatorOutput)
        
        # Build the workflow graph
        await self.build_graph()

    # Worker Node Logic
    def worker(self, state: State) -> Dict[str, Any]:
        """
        Main worker function. Receives the state, constructs a system message with instructions
        and tools, then invokes the worker LLM.
        """
        # System instructions for the assistant
        system_message = f"""You are a helpful assistant that can use tools to complete tasks.
You keep working on a task until either you have a question or clarification for the user, or the success criteria is met.
You have many tools to help you, including tools to browse the internet, navigating and retrieving web pages.
You have a tool to run python code, but note that you would need to include a print() statement if you wanted to receive output.
The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This is the success criteria:
{state["success_criteria"]}
You should reply either with a question for the user about this assignment, or with your final response.
If you have a question for the user, you need to reply by clearly stating your question. An example might be:

Question: please clarify whether you want a summary or a detailed answer

If you've finished, reply with the final answer, and don't ask a question; simply reply with the answer.
"""

        # Include feedback from previous failed attempt, if any
        if state.get("feedback_on_work"):
            system_message += f"""
Previously you thought you completed the assignment, but your reply was rejected because the success criteria was not met.
Here is the feedback on why this was rejected:
{state["feedback_on_work"]}
With this feedback, please continue the assignment, ensuring that you meet the success criteria or have a question for the user."""

        # Add system message to the chat history
        found_system_message = False
        messages = state["messages"]
        for message in messages:
            if isinstance(message, SystemMessage):
                message.content = system_message
                found_system_message = True
        if not found_system_message:
            messages = [SystemMessage(content=system_message)] + messages

        # Invoke worker LLM with the messages
        response = self.worker_llm_with_tools.invoke(messages)

        # Return updated state
        return {"messages": [response]}

    # Worker Router
    def worker_router(self, state: State) -> str:
        """
        Decide whether the worker's last response should go to a tool node
        or directly to the evaluator.
        """
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        else:
            return "evaluator"

    # Format conversation for evaluator
    def format_conversation(self, messages: List[Any]) -> str:
        """Convert chat history into a readable string for the evaluator"""
        conversation = "Conversation history:\n\n"
        for message in messages:
            if isinstance(message, HumanMessage):
                conversation += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                text = message.content or "[Tools use]"
                conversation += f"Assistant: {text}\n"
        return conversation

    # Evaluator Node Logic
    def evaluator(self, state: State) -> State:
        """
        Evaluator LLM checks if the assistant's response meets success criteria,
        provides feedback, and indicates if user input is needed.
        """
        last_response = state["messages"][-1].content

        # System instructions for evaluator
        system_message = """You are an evaluator that determines if a task has been completed successfully by an Assistant.
Assess the Assistant's last response based on the given criteria. Respond with your feedback, and with your decision on whether the success criteria has been met,
and whether more input is needed from the user."""

        # User message for evaluator
        user_message = f"""You are evaluating a conversation between the User and Assistant. You decide what action to take based on the last response from the Assistant.

The entire conversation with the assistant, with the user's original request and all replies, is:
{self.format_conversation(state["messages"])}

The success criteria for this assignment is:
{state["success_criteria"]}

And the final response from the Assistant that you are evaluating is:
{last_response}

Respond with your feedback, and decide if the success criteria is met by this response.
Also, decide if more user input is required, either because the assistant has a question, needs clarification, or seems to be stuck and unable to answer without help.

The Assistant has access to a tool to write files. If the Assistant says they have written a file, then you can assume they have done so.
Overall you should give the Assistant the benefit of the doubt if they say they've done something. But you should reject if you feel that more work should go into this.
"""
        # Include prior feedback
        if state["feedback_on_work"]:
            user_message += f"Also, note that in a prior attempt from the Assistant, you provided this feedback: {state['feedback_on_work']}\n"
            user_message += "If you're seeing the Assistant repeating the same mistakes, then consider responding that user input is required."

        evaluator_messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message),
        ]

        # Invoke evaluator LLM with structured output
        eval_result = self.evaluator_llm_with_output.invoke(evaluator_messages)

        # Return updated state with feedback
        new_state = {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Evaluator Feedback on this answer: {eval_result.feedback}",
                }
            ],
            "feedback_on_work": eval_result.feedback,
            "success_criteria_met": eval_result.success_criteria_met,
            "user_input_needed": eval_result.user_input_needed,
        }
        return new_state

    # Evaluation Routing
    def route_based_on_evaluation(self, state: State) -> str:
        """Determine next node based on evaluation results"""
        if state["success_criteria_met"] or state["user_input_needed"]:
            return "END"
        else:
            return "worker"

    # Build Workflow Graph
    async def build_graph(self):
        """
        Construct the workflow graph connecting worker, tools, and evaluator nodes,
        with conditional edges and checkpointing.
        """
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("worker", self.worker)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_node("evaluator", self.evaluator)

        # Add conditional edges
        graph_builder.add_conditional_edges(
            "worker", self.worker_router, {"tools": "tools", "evaluator": "evaluator"}
        )
        graph_builder.add_edge("tools", "worker")
        graph_builder.add_conditional_edges(
            "evaluator", self.route_based_on_evaluation, {"worker": "worker", "END": END}
        )
        graph_builder.add_edge(START, "worker")

        # Compile the graph with memory checkpointer
        self.graph = graph_builder.compile(checkpointer=self.memory)

    # Run a Superstep
    async def run_superstep(self, message, success_criteria, history):
        """
        Executes one "superstep" of the graph, processing user input through
        worker, tools, and evaluator nodes, and returns updated chat history.
        """
        config = {"configurable": {"thread_id": self.sidekick_id}}

        # Initialize state for this step
        state = {
            "messages": message,
            "success_criteria": success_criteria or "The answer should be clear and accurate",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
        }

        # Run graph asynchronously
        result = await self.graph.ainvoke(state, config=config)

        # Format chat messages for history
        user = {"role": "user", "content": message}
        reply = {"role": "assistant", "content": result["messages"][-2].content}
        feedback = {"role": "assistant", "content": result["messages"][-1].content}
        return history + [user, reply, feedback]

    # Cleanup Resources
    def cleanup(self):
        """
        Close Playwright browser and stop Playwright instance.
        Handles both async loop running and direct execution.
        """
        if self.browser:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.browser.close())
                if self.playwright:
                    loop.create_task(self.playwright.stop())
            except RuntimeError:
                # If no loop is running, close directly
                asyncio.run(self.browser.close())
                if self.playwright:
                    asyncio.run(self.playwright.stop())

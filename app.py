# sidekick_gradio.py
# This script creates a Gradio interface for interacting with the Sidekick AI.
# It handles setup, user messages, workflow execution, resetting, and resource cleanup.

import gradio as gr
from sidekick import Sidekick  # Your autonomous assistant class


# Setup Sidekick
async def setup():
    """
    Initialize a new Sidekick instance asynchronously.
    Returns the Sidekick object to store in Gradio state.
    """
    sidekick = Sidekick()
    await sidekick.setup()
    return sidekick


# Process User Message
async def process_message(sidekick, message, success_criteria, history):
    """
    Sends the user's message to the Sidekick and runs one superstep of the workflow.
    
    Parameters:
        sidekick: The Sidekick instance
        message: User's input message
        success_criteria: Text describing success criteria for this task
        history: Chat history (list of messages)
    
    Returns:
        Updated chat history and Sidekick instance
    """
    results = await sidekick.run_superstep(message, success_criteria, history)
    return results, sidekick


# Reset Sidekick
async def reset():
    """
    Resets the chat by creating a fresh Sidekick instance.
    Returns empty fields and the new Sidekick instance.
    """
    new_sidekick = Sidekick()
    await new_sidekick.setup()
    return "", "", None, new_sidekick


# Cleanup Resources
def free_resources(sidekick):
    """
    Called when the Gradio state is deleted.
    Ensures any open browsers or Playwright processes are closed.
    """
    print("Cleaning up")
    try:
        if sidekick:
            sidekick.cleanup()
    except Exception as e:
        print(f"Exception during cleanup: {e}")


# Build Gradio Interface

with gr.Blocks(title="Sidekick", theme=gr.themes.Default(primary_hue="emerald")) as ui:
    # Title
    gr.Markdown("## Sidekick Personal Co-Worker")

    # Store Sidekick instance in state
    sidekick = gr.State(delete_callback=free_resources)

    # Chatbot output
    with gr.Row():
        chatbot = gr.Chatbot(label="Sidekick", height=300, type="messages")

    # Input fields
    with gr.Group():
        with gr.Row():
            message = gr.Textbox(show_label=False, placeholder="Your request to the Sidekick")
        with gr.Row():
            success_criteria = gr.Textbox(
                show_label=False, placeholder="What are your success criteria?"
            )

    # Buttons
    with gr.Row():
        reset_button = gr.Button("Reset", variant="stop")
        go_button = gr.Button("Go!", variant="primary")

    # Load event: initialize Sidekick when UI loads
    ui.load(setup, [], [sidekick])

    # Submit events: enter key or button triggers message processing
    message.submit(
        process_message, [sidekick, message, success_criteria, chatbot], [chatbot, sidekick]
    )
    success_criteria.submit(
        process_message, [sidekick, message, success_criteria, chatbot], [chatbot, sidekick]
    )
    go_button.click(
        process_message, [sidekick, message, success_criteria, chatbot], [chatbot, sidekick]
    )

    # Reset button
    reset_button.click(reset, [], [message, success_criteria, chatbot, sidekick])


# Launch the Gradio interface in the browser
ui.launch(inbrowser=True)

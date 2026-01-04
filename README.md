# Sidekick AI - Personal Co-Worker

## Overview

Sidekick is an AI-powered personal co-worker built with LangChain and Gradio. It integrates multiple tools, including web browsing, file management, Wikipedia, Python code execution, and push notifications. The AI assistant is capable of iterative self-correction, executing workflows, and providing detailed responses based on user-defined success criteria.

## Features

* **Tool Integration:** Access web searches, Wikipedia, Python execution, file management, and push notifications.
* **Asynchronous Setup:** Efficient initialization of browser and tool instances.
* **Evaluator Feedback Loops:** Iterative self-correction to ensure output meets success criteria.
* **Gradio Interface:** Clean web UI with isolated user sessions, chat history, and dynamic input fields.
* **Resource Management:** Automatic cleanup of Playwright browser instances.

## Files

* `app.py`: Gradio interface for interacting with the Sidekick AI.
* `sidekick.py`: Core AI logic, including workflow, evaluation, and tool integration.
* `sidekick_tools.py`: Implements all external tools and API integrations.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:

```env
PUSHOVER_TOKEN=your_token_here
PUSHOVER_USER=your_user_key_here
```

## Usage

1. Launch the Gradio app:

```bash
python app.py
```

2. Open the local URL displayed in your browser.
3. Type a request and optional success criteria.
4. Click **Go!** to send the message to the Sidekick.
5. Use **Reset** to start a new session.

## Contributing

* Add more tools in `sidekick_tools.py` and they will be automatically available to the Sidekick.
* Improve prompts, evaluation criteria, and workflows in `sidekick.py`.
* Ensure proper cleanup in asynchronous methods to avoid resource leaks.


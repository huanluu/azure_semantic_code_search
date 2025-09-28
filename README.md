# Agentic Retrieval Pipeline (python3 Script)

This folder contains a runnable python3 project that mirrors the workflow from the `agent-example.ipynb` notebook in the Azure Search python3 samples. It provisions Azure AI Search resources, wires them into an agentic retrieval pipeline with Azure AI Agent Service, runs sample conversations, and (optionally) cleans up resources — all from the command line.

## Prerequisites

- python3 3.10 or later (tested with python3 3.13).
- Access to the following Azure resources in the same tenant:
  - Azure AI Search service (basic tier or higher) with semantic ranker enabled.
  - Azure OpenAI resource with deployed chat and embedding models (for example `gpt-4o-mini` and `text-embedding-3-large`).
  - Azure AI Foundry project with agent tooling enabled.
- The preview Azure SDKs referenced in `requirements.txt` (install them via `pip`).

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   ```fish
   pip install -r requirements.txt
   ```

3. Copy `.env.sample` to `.env` and fill in the values for your environment. The script uses Azure AD / MSI authentication through `DefaultAzureCredential`, so ensure your identity has sufficient permissions on all services.

## Usage

The entry-point script `azure_agent_pipeline.py` offers sub-commands to run the full demo or individual stages.

```fish
python3 azure_agent_pipeline.py --help
```

### Full demo (create resources, run two questions, print retrieval details)

```fish
python3 azure_agent_pipeline.py demo
```

### Provision Azure AI Search resources only

```fish
python3 azure_agent_pipeline.py setup
```

### Run the primary sample question (assumes `setup` already ran)

```fish
python3 azure_agent_pipeline.py ask --prompt "Why do suburban belts display larger December brightening than urban cores even though absolute light levels are higher downtown?"
```

### Review retrieval traces for the most recent question

```fish
python3 azure_agent_pipeline.py review
```

### Clean up the created resources

```fish
python3 azure_agent_pipeline.py cleanup
```

## Environment variables

Refer to `.env.sample` for the complete list. Key values include:

- `PROJECT_ENDPOINT`
- `AZURE_SEARCH_ENDPOINT`
- `AZURE_SEARCH_INDEX`
- `AZURE_SEARCH_KNOWLEDGE_SOURCE_NAME`
- `AZURE_SEARCH_AGENT_NAME`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_GPT_DEPLOYMENT`
- `AZURE_OPENAI_GPT_MODEL`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
- `AZURE_OPENAI_EMBEDDING_MODEL`
- `AGENT_MODEL`

## Notes

- The script relies on preview API features that may change.
- For production use, add richer logging, retry logic, and idempotency guards as needed.
- Clean up resources when you are done to avoid unwanted charges.

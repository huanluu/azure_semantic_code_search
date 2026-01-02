# Build and Test Guide

This document describes how to build, test, and verify the Azure Semantic Code Search project.

## Prerequisites

- **Python**: 3.10 or later (tested with Python 3.12)
- **pip**: Python package installer
- **Virtual environment**: Recommended for dependency isolation

## Build Steps

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate     # On Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `azure-search-documents` (11.7.0b1) - Azure AI Search SDK
- `azure-ai-projects` (>=1.0.0b8) - Azure AI Project SDK
- `azure-ai-agents` (>=1.0.0b6) - Azure AI Agents SDK
- `azure-identity` (>=1.17.0) - Azure authentication
- `python-dotenv` (>=1.0.1) - Environment variable management
- `requests` (>=2.32.3) - HTTP library

### 3. Verify Build

Run the smoke test to verify the build:

```bash
python3 test_build.py
```

Expected output:
```
======================================================================
Azure Semantic Code Search - Build Smoke Test
======================================================================

Testing Python syntax...
  ✓ Syntax validation passed

Testing imports...
  ✓ All imports successful

Testing configuration loading...
  ✓ Configuration loading passed
    - Index name: earth_at_night
    - Agent model: gpt-4.1-mini

Testing Swift file access...
  ✓ Swift file access passed
    - Found 23 Swift files
    - Sample file: FluentTheme+Tokens.swift (10151 bytes)

Testing CLI argument parser...
  ✓ CLI argument parser passed
    - Available commands: demo, setup, ask, review, cleanup

======================================================================
✓ All 5 tests passed!
======================================================================
```

### 4. Verify CLI

Test the command-line interface:

```bash
python3 azure_agent_pipeline.py --help
```

You should see the help message with available commands:
- `demo` - Run the end-to-end demo flow
- `setup` - Provision Azure AI Search assets
- `ask` - Ask a custom question
- `review` - Review retrieval traces
- `cleanup` - Clean up resources

## Project Structure

```
azure_semantic_code_search/
├── azure_agent_pipeline.py    # Main application script
├── requirements.txt            # Python dependencies
├── test_build.py              # Build smoke tests
├── BUILD.md                   # This file
├── README.md                  # Project documentation
├── .env.sample                # Sample environment configuration
├── .gitignore                 # Git ignore rules
└── fluentui-apple-code/       # Swift source files for indexing
    └── FluentUI_common/       # Sample Swift code
```

## Troubleshooting

### Import Errors

If you encounter import errors, ensure you:
1. Activated the virtual environment
2. Installed all dependencies from requirements.txt
3. Are using Python 3.10 or later

### Missing Swift Files

The project expects Swift source files in `fluentui-apple-code/` directory. The smoke test verifies these files are accessible.

### Azure Credential Issues

The application uses `DefaultAzureCredential` for Azure authentication. To run the full application (not just build tests), you need:
1. Valid Azure credentials
2. Properly configured `.env` file (copy from `.env.sample`)
3. Access to required Azure resources (AI Search, OpenAI, AI Foundry)

## Continuous Integration

The smoke test (`test_build.py`) can be integrated into CI/CD pipelines to verify builds:

```bash
# In CI environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 test_build.py
```

Exit code:
- `0` - All tests passed
- `1` - One or more tests failed

## Next Steps

After verifying the build:
1. Configure your `.env` file with Azure credentials
2. Run `python3 azure_agent_pipeline.py setup` to provision resources
3. Use `python3 azure_agent_pipeline.py ask --prompt "your question"` to test the pipeline

For more information, see [README.md](README.md).

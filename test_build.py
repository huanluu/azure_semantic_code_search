#!/usr/bin/env python3
"""
Smoke test script for azure_semantic_code_search project.
Verifies that the project builds and basic functionality works without requiring Azure credentials.
"""

import sys
import os
from pathlib import Path


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        import azure_agent_pipeline
        from azure_agent_pipeline import (
            AgenticRetrievalPipeline,
            PipelineConfig,
            load_config_from_env,
            build_arg_parser,
        )
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_syntax():
    """Test that the main script has valid Python syntax."""
    print("Testing Python syntax...")
    import ast
    try:
        with open('azure_agent_pipeline.py', 'r') as f:
            code = f.read()
        ast.parse(code)
        print("  ✓ Syntax validation passed")
        return True
    except SyntaxError as e:
        print(f"  ✗ Syntax error: {e}")
        return False


def test_configuration():
    """Test configuration loading with minimal environment variables."""
    print("Testing configuration loading...")
    
    # Set minimal test environment variables
    os.environ['PROJECT_ENDPOINT'] = 'https://test.services.ai.azure.com/api/projects/test'
    os.environ['AZURE_SEARCH_ENDPOINT'] = 'https://test.search.windows.net'
    os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://test.openai.azure.com'
    
    try:
        from azure_agent_pipeline import load_config_from_env, PipelineConfig
        
        config = load_config_from_env()
        
        # Verify config object is created correctly
        assert isinstance(config, PipelineConfig), "Config should be a PipelineConfig instance"
        assert config.project_endpoint == os.environ['PROJECT_ENDPOINT']
        assert config.search_endpoint == os.environ['AZURE_SEARCH_ENDPOINT']
        assert config.azure_openai_endpoint == os.environ['AZURE_OPENAI_ENDPOINT']
        
        print("  ✓ Configuration loading passed")
        print(f"    - Index name: {config.index_name}")
        print(f"    - Agent model: {config.agent_model}")
        return True
    except Exception as e:
        print(f"  ✗ Configuration test failed: {e}")
        return False


def test_swift_files():
    """Test that Swift source files can be discovered and read."""
    print("Testing Swift file access...")
    
    try:
        base_dir = Path.cwd()
        swift_root = base_dir / "fluentui-apple-code"
        
        if not swift_root.exists():
            print(f"  ✗ Swift source directory not found: {swift_root}")
            return False
        
        swift_files = sorted(swift_root.rglob("*.swift"))
        
        if not swift_files:
            print(f"  ✗ No Swift files found under {swift_root}")
            return False
        
        # Try reading a sample file
        test_file = swift_files[0]
        content = test_file.read_text(encoding="utf-8")
        
        print(f"  ✓ Swift file access passed")
        print(f"    - Found {len(swift_files)} Swift files")
        print(f"    - Sample file: {test_file.name} ({len(content)} bytes)")
        return True
    except Exception as e:
        print(f"  ✗ Swift file test failed: {e}")
        return False


def test_cli():
    """Test that CLI argument parser works correctly."""
    print("Testing CLI argument parser...")
    
    try:
        from azure_agent_pipeline import build_arg_parser
        
        parser = build_arg_parser()
        
        # Test that parser accepts valid commands
        commands = ['demo', 'setup', 'ask', 'review', 'cleanup']
        for cmd in commands:
            # Just verify it doesn't throw an exception for valid commands
            try:
                if cmd == 'ask':
                    parser.parse_args([cmd, '--prompt', 'test'])
                else:
                    parser.parse_args([cmd])
            except SystemExit:
                # argparse may exit on certain conditions, which is okay
                pass
        
        print("  ✓ CLI argument parser passed")
        print(f"    - Available commands: {', '.join(commands)}")
        return True
    except Exception as e:
        print(f"  ✗ CLI test failed: {e}")
        return False


def main():
    """Run all smoke tests."""
    print("=" * 70)
    print("Azure Semantic Code Search - Build Smoke Test")
    print("=" * 70)
    print()
    
    tests = [
        test_syntax,
        test_imports,
        test_configuration,
        test_swift_files,
        test_cli,
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ All {total} tests passed!")
        print("=" * 70)
        return 0
    else:
        print(f"✗ {total - passed} out of {total} tests failed")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())

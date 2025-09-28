"""Agentic retrieval pipeline scripted version of the agent-example notebook.

Run this module as a CLI to provision Azure AI Search assets, wire them to an Azure AI
Agent, execute sample questions, inspect retrieval traces, and optionally clean up.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
import hashlib
from typing import Dict, Iterable, Optional, Any

from pathlib import Path
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.agent import KnowledgeAgentRetrievalClient
from azure.search.documents.agent.models import (
    KnowledgeAgentMessage,
    KnowledgeAgentMessageTextContent,
    KnowledgeAgentRetrievalRequest,
)
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    HnswAlgorithmConfiguration,
    KnowledgeAgent,
    KnowledgeAgentAzureOpenAIModel,
    KnowledgeAgentOutputConfiguration,
    KnowledgeAgentOutputConfigurationModality,
    KnowledgeSourceReference,
    SearchField,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    VectorSearch,
    VectorSearchProfile,
)
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import (
    AgentsNamedToolChoice,
    AgentsNamedToolChoiceType,
    FunctionName,
    FunctionTool,
    ListSortOrder,
    ToolSet,
)
from dotenv import load_dotenv


@dataclass
class PipelineConfig:
    project_endpoint: str
    search_endpoint: str
    azure_openai_endpoint: str
    azure_openai_gpt_deployment: str
    azure_openai_gpt_model: str
    azure_openai_embedding_deployment: str
    azure_openai_embedding_model: str
    index_name: str
    knowledge_source_name: str
    agent_name: str
    agent_model: str
    project_agent_name: str
    documents_url: str = (
        "https://raw.githubusercontent.com/Azure-Samples/azure-search-sample-data/refs/heads/main/"
        "nasa-e-book/earth-at-night-json/documents.json"
    )


class AgenticRetrievalPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.credential = DefaultAzureCredential()
        self.index_client = SearchIndexClient(endpoint=config.search_endpoint, credential=self.credential)
        self.project_client = AIProjectClient(endpoint=config.project_endpoint, credential=self.credential)
        self.agent_client: Optional[KnowledgeAgentRetrievalClient] = None
        self.thread = None
        self.toolset: Optional[ToolSet] = None
        self.retrieval_results: Dict[str, Any] = {}
        self.agent = None
        self.last_message_id: Optional[str] = None

    # region Provisioning helpers
    def create_search_index(self) -> None:
        cfg = self.config
        print(f"Creating or updating search index '{cfg.index_name}'...")
        index = SearchIndex(
            name=cfg.index_name,
            fields=[
                SearchField(name="id", type="Edm.String", key=True, filterable=True, sortable=True, facetable=True),
                SearchField(name="page_chunk", type="Edm.String", filterable=False, sortable=False, facetable=False),
                SearchField(
                    name="page_embedding_text_3_large",
                    type="Collection(Edm.Single)",
                    stored=False,
                    vector_search_dimensions=3072,
                    vector_search_profile_name="hnsw_text_3_large",
                ),
                SearchField(name="page_number", type="Edm.Int32", filterable=True, sortable=True, facetable=True),
            ],
            vector_search=VectorSearch(
                profiles=[
                    VectorSearchProfile(
                        name="hnsw_text_3_large",
                        algorithm_configuration_name="alg",
                        vectorizer_name="azure_openai_text_3_large",
                    )
                ],
                algorithms=[HnswAlgorithmConfiguration(name="alg")],
                vectorizers=[
                    AzureOpenAIVectorizer(
                        vectorizer_name="azure_openai_text_3_large",
                        parameters=AzureOpenAIVectorizerParameters(
                            resource_url=cfg.azure_openai_endpoint,
                            deployment_name=cfg.azure_openai_embedding_deployment,
                            model_name=cfg.azure_openai_embedding_model,
                        ),
                    )
                ],
            ),
            semantic_search=SemanticSearch(
                default_configuration_name="semantic_config",
                configurations=[
                    SemanticConfiguration(
                        name="semantic_config",
                        prioritized_fields=SemanticPrioritizedFields(
                            content_fields=[SemanticField(field_name="page_chunk")]
                        ),
                    )
                ],
            ),
        )
        self.index_client.create_or_update_index(index)
        print(f"Index '{cfg.index_name}' is ready.")

    def upload_documents(self) -> None:
        cfg = self.config
        print("Collecting Swift source files for upload...")
        base_dir = Path(__file__).resolve().parent
        swift_root = base_dir / "fluentui-apple-code"
        if not swift_root.exists():
            raise FileNotFoundError(f"Swift source directory not found: {swift_root}")

        swift_files = sorted(swift_root.rglob("*.swift"))
        if not swift_files:
            raise RuntimeError(f"No Swift files found under {swift_root}.")

        documents = []
        for idx, file_path in enumerate(swift_files, start=1):
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = file_path.read_text(encoding="utf-8", errors="ignore")

            relative_path = file_path.relative_to(swift_root).as_posix()
            page_chunk = f"// File: {relative_path}\n\n{content}"
            doc_id = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()
            documents.append(
                {
                    "id": doc_id,
                    "page_chunk": page_chunk,
                    "page_number": idx,
                }
            )

        print(f"Uploading {len(documents)} Swift documents to '{cfg.index_name}'...")
        search_client = SearchClient(
            endpoint=cfg.search_endpoint,
            index_name=cfg.index_name,
            credential=self.credential,
        )
        results = search_client.upload_documents(documents=documents)
        failed = [r for r in results if not r.succeeded]
        if failed:
            failed_keys = ", ".join(result.key for result in failed)
            raise RuntimeError(
                f"Failed to index {len(failed)} documents. Keys: {failed_keys}"
            )
        print(f"Uploaded {len(documents)} documents to '{cfg.index_name}'.")

    def create_knowledge_source(self) -> None:
        cfg = self.config
        print(f"Creating or updating knowledge source '{cfg.knowledge_source_name}'...")
        try:
            from azure.search.documents.indexes.models import (
                SearchIndexKnowledgeSource,
                SearchIndexKnowledgeSourceParameters,
            )
        except (ImportError, AttributeError):
            from azure.search.documents.indexes._generated.models import (
                SearchIndexKnowledgeSource,  # type: ignore[attr-defined]
                SearchIndexKnowledgeSourceParameters,  # type: ignore[attr-defined]
            )
        ks = SearchIndexKnowledgeSource(
            name=cfg.knowledge_source_name,
            description="Knowledge source for Earth at night data",
            search_index_parameters=SearchIndexKnowledgeSourceParameters(
                search_index_name=cfg.index_name,
                source_data_select="id,page_chunk,page_number",
            ),
        )
        self.index_client.create_or_update_knowledge_source(knowledge_source=ks)
        print(f"Knowledge source '{cfg.knowledge_source_name}' is ready.")

    def create_knowledge_agent(self) -> None:
        cfg = self.config
        print(f"Creating or updating knowledge agent '{cfg.agent_name}'...")
        aoai_params = AzureOpenAIVectorizerParameters(
            resource_url=cfg.azure_openai_endpoint,
            deployment_name=cfg.azure_openai_gpt_deployment,
            model_name=cfg.azure_openai_gpt_model,
        )
        output_cfg = KnowledgeAgentOutputConfiguration(
            modality=KnowledgeAgentOutputConfigurationModality.EXTRACTIVE_DATA,
            include_activity=True,
        )
        agent = KnowledgeAgent(
            name=cfg.agent_name,
            models=[KnowledgeAgentAzureOpenAIModel(azure_open_ai_parameters=aoai_params)],
            knowledge_sources=[
                KnowledgeSourceReference(
                    name=cfg.knowledge_source_name,
                    include_reference_source_data=True,
                    include_references=True,
                )
            ],
            output_configuration=output_cfg,
        )
        self.index_client.create_or_update_agent(agent)
        print(f"Knowledge agent '{cfg.agent_name}' is ready.")

    def ensure_ai_agent(self) -> None:
        cfg = self.config
        print(f"Creating Foundry agent '{cfg.project_agent_name}'...")
        instructions = (
            "A Q&A agent that can answer questions about the Earth at night.\n"
            "Sources have a JSON format with a ref_id that must be cited in the answer using the format [ref_id].\n"
            "If you do not have the answer, respond with \"I don't know\"."
        )
        self.agent = self.project_client.agents.create_agent(
            model=cfg.agent_model,
            name=cfg.project_agent_name,
            instructions=instructions,
        )
        print(f"AI agent '{cfg.project_agent_name}' is ready.")

    def configure_agentic_tool(self) -> None:
        cfg = self.config
        if self.agent is None:
            raise RuntimeError("Call ensure_ai_agent() before configure_agentic_tool().")

        print("Configuring agentic retrieval toolset...")
        self.agent_client = KnowledgeAgentRetrievalClient(
            endpoint=cfg.search_endpoint, agent_name=cfg.agent_name, credential=self.credential
        )
        self.thread = self.project_client.agents.threads.create()
        self.retrieval_results = {}

        def agentic_retrieval() -> str:
            """Retrieve context for the latest user message via Azure AI Search."""
            assert self.thread is not None
            if self.agent_client is None:
                raise RuntimeError("KnowledgeAgentRetrievalClient is not initialized.")

            messages: Iterable[dict] = self.project_client.agents.messages.list(
                self.thread.id, limit=5, order=ListSortOrder.DESCENDING
            )
            ordered = list(messages)
            ordered.reverse()
            retrieval_result = self.agent_client.retrieve(
                retrieval_request=KnowledgeAgentRetrievalRequest(
                    messages=[
                        KnowledgeAgentMessage(
                            role=m["role"],
                            content=[KnowledgeAgentMessageTextContent(text=m["content"])],
                        )
                        for m in ordered
                        if m["role"] != "system"
                    ]
                )
            )
            last_message = ordered[-1]
            self.retrieval_results[last_message.id] = retrieval_result
            # Return unified string to the agent
            return retrieval_result.response[0].content[0].text

        functions = FunctionTool({agentic_retrieval})
        self.toolset = ToolSet()
        self.toolset.add(functions)
        self.project_client.agents.enable_auto_function_calls(self.toolset)
        print("Function calling is enabled.")

    # endregion

    # region Conversations
    def ask_question(self, prompt: str, review: bool = False) -> str:
        # breakpoint()
        if self.agent is None or self.thread is None or self.toolset is None:
            raise RuntimeError("Call setup_pipeline() before asking questions.")

        print(f"Sending question: {prompt.strip()[:80]}...")
        message = self.project_client.agents.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=prompt,
        )
        run = self.project_client.agents.runs.create_and_process(
            thread_id=self.thread.id,
            agent_id=self.agent.id,
            tool_choice=AgentsNamedToolChoice(
                type=AgentsNamedToolChoiceType.FUNCTION,
                function=FunctionName(name="agentic_retrieval"),
            ),
            toolset=self.toolset,
        )
        if run["status"] == "failed":
            raise RuntimeError(f"Run failed: {run['last_error']}")
        output = self.project_client.agents.messages.get_last_message_text_by_role(
            thread_id=self.thread.id, role="assistant"
        ).text.value
        print("Agent response: \n" + output)
        self.last_message_id = message.id
        if review:
            self.review_retrieval(message.id)
        return message.id

    def review_retrieval(self, message_id: str) -> None:
        retrieval_result = self.retrieval_results.get(message_id)
        if retrieval_result is None:
            raise RuntimeError(f"No retrieval results cached for message {message_id}.")
        print("\nRetrieval activity:")
        print(json.dumps([activity.as_dict() for activity in retrieval_result.activity], indent=2))
        print("\nRetrieval results:")
        print(json.dumps([reference.as_dict() for reference in retrieval_result.references], indent=2))

    # endregion

    # region Orchestration helpers
    def _search_index_exists(self) -> bool:
        try:
            self.index_client.get_index(self.config.index_name)
            return True
        except ResourceNotFoundError:
            return False

    def _knowledge_source_exists(self) -> bool:
        try:
            self.index_client.get_knowledge_source(self.config.knowledge_source_name)
            return True
        except ResourceNotFoundError:
            return False
        except HttpResponseError as err:
            if getattr(err, "status_code", None) == 404:
                return False
            raise

    def _knowledge_agent_exists(self) -> bool:
        try:
            self.index_client.get_agent(self.config.agent_name)
            return True
        except ResourceNotFoundError:
            return False
        except HttpResponseError as err:
            if getattr(err, "status_code", None) == 404:
                return False
            raise

    def _get_existing_ai_agent(self) -> Optional[Any]:
        try:
            for agent in self.project_client.agents.list_agents():
                if getattr(agent, "name", None) == self.config.project_agent_name:
                    return agent
        except HttpResponseError as err:
            if getattr(err, "status_code", None) != 404:
                raise
        return None

    def setup_pipeline(self, upload_docs: bool = True) -> None:
        cfg = self.config

        if self._search_index_exists():
            print(f"Search index '{cfg.index_name}' already exists. Skipping creation, assusing doc already uploaded.")
        else:
            self.create_search_index()
            if upload_docs:
                self.upload_documents()
        if self._knowledge_source_exists():
            print(f"Knowledge source '{cfg.knowledge_source_name}' already exists. Skipping creation.")
        else:
            self.create_knowledge_source()
        if self._knowledge_agent_exists():
            print(f"Knowledge agent '{cfg.agent_name}' already exists. Skipping creation.")
        else:
            self.create_knowledge_agent()
        existing_agent = self._get_existing_ai_agent()
        if existing_agent is not None:
            print(f"AI agent '{cfg.project_agent_name}' already exists. Reusing existing agent.")
            self.agent = existing_agent
        else:
            self.ensure_ai_agent()
        self.configure_agentic_tool()



    def run_demo(self, include_cleanup: bool = False) -> None:
        self.setup_pipeline()
        primary_question = (
            "Why do suburban belts display larger December brightening than urban cores even though "
            "absolute light levels are higher downtown?"
        )
        secondary_question = (
            "Why is the Phoenix nighttime street grid so sharply visible from space, whereas large stretches "
            "of the interstate between midwestern cities remain comparatively dim?"
        )
        self.ask_question(primary_question, review=True)
        self.ask_question(secondary_question, review=True)
        if include_cleanup:
            self.cleanup()

    def cleanup(self) -> None:
        cfg = self.config
        print("Cleaning up resources...")
        try:
            self.index_client.delete_agent(cfg.agent_name)
            print(f"Deleted knowledge agent '{cfg.agent_name}'.")
        except ResourceNotFoundError:
            print("Knowledge agent already deleted or missing.")
        try:
            existing_agent = self._get_existing_ai_agent()
            if existing_agent is not None:
                agent_id = getattr(existing_agent, "id", None)
                if agent_id:
                    delete_agent = getattr(self.project_client.agents, "delete_agent", None)
                    begin_delete_agent = getattr(self.project_client.agents, "begin_delete_agent", None)
                    if callable(delete_agent):
                        delete_agent(agent_id)
                    elif callable(begin_delete_agent):
                        poller = begin_delete_agent(agent_id)
                        try:
                            poller.wait()
                        except AttributeError:
                            # Fallback for pollers without wait(); best effort completion.
                            pass
                    else:
                        print("Foundry agent delete API unavailable; skipping delete.")
                    print(f"Deleted Foundry agent '{cfg.project_agent_name}'.")
                else:
                    print("Foundry agent record missing identifier; skipping delete.")
            else:
                print("Foundry agent already deleted or missing.")
        except (HttpResponseError, AttributeError) as err:
            print(f"Foundry agent cleanup skipped: {err}.")
        try:
            self.index_client.delete_knowledge_source(knowledge_source=cfg.knowledge_source_name)
            print(f"Deleted knowledge source '{cfg.knowledge_source_name}'.")
        except (ResourceNotFoundError, HttpResponseError) as err:
            print(f"Knowledge source cleanup skipped: {err}.")
        try:
            self.index_client.delete_index(cfg.index_name)
            print(f"Deleted index '{cfg.index_name}'.")
        except ResourceNotFoundError:
            print("Index already deleted or missing.")
        self.agent = None
        self.thread = None
        self.toolset = None
        print("Cleanup complete.")

    # endregion


def load_config_from_env() -> PipelineConfig:
    load_dotenv(override=True)
    required_vars = [
        "PROJECT_ENDPOINT",
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_OPENAI_ENDPOINT",
    ]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")
    agent_name = os.getenv("AZURE_SEARCH_AGENT_NAME", "earth-search-agent")
    project_agent_name = os.getenv("PROJECT_AGENT_NAME", f"{agent_name}-orchestrator")
    return PipelineConfig(
        project_endpoint=os.environ["PROJECT_ENDPOINT"],
        search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        azure_openai_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_openai_gpt_deployment=os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT", "gpt-4.1-mini"),
        azure_openai_gpt_model=os.getenv("AZURE_OPENAI_GPT_MODEL", "gpt-4.1-mini"),
        azure_openai_embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"),
        azure_openai_embedding_model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        index_name=os.getenv("AZURE_SEARCH_INDEX", "earth_at_night"),
        knowledge_source_name=os.getenv("AZURE_SEARCH_KNOWLEDGE_SOURCE_NAME", "earth-at-night-ks"),
        agent_name=agent_name,
        agent_model=os.getenv("AGENT_MODEL", "gpt-4.1-mini"),
        project_agent_name=project_agent_name,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic retrieval pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="Run the end-to-end demo flow.")
    demo_parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete created resources after the demo completes.",
    )

    subparsers.add_parser("setup", help="Provision Azure AI Search assets and configure the Foundry agent.")

    ask_parser = subparsers.add_parser("ask", help="Ask a custom question using the pipeline.")
    ask_parser.add_argument("--prompt", required=True, help="Prompt to send to the agent.")
    ask_parser.add_argument(
        "--review",
        action="store_true",
        help="Print retrieval traces for the generated answer.",
    )

    review_parser = subparsers.add_parser(
        "review", help="Run the default sample question and print retrieval traces."
    )
    review_parser.add_argument(
        "--prompt",
        default=(
            "Why do suburban belts display larger December brightening than urban cores even though "
            "absolute light levels are higher downtown?"
        ),
        help="Prompt to use when reviewing retrieval output.",
    )

    subparsers.add_parser("cleanup", help="Delete the created search index, knowledge source, and agents.")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config_from_env()
    pipeline = AgenticRetrievalPipeline(config)

    if args.command == "demo":
        pipeline.run_demo(include_cleanup=args.cleanup)
    elif args.command == "setup":
        pipeline.setup_pipeline()
        # Enter interactive mode
        print("\nSetup complete! Entering interactive mode...")
        print("Commands: 'ask <question>', 'review [question]', 'quit'")
        print("  - 'ask <question>': Ask a new question")
        print("  - 'review': Show retrieval details for the last question")
        print("  - 'review <question>': Ask a new question and show retrieval details")
        print("  - 'quit': Exit and cleanup")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                    
                elif user_input.startswith('ask '):
                    question = user_input[4:].strip()
                    if question:
                        pipeline.ask_question(question, review=False)
                    else:
                        print("Please provide a question after 'ask'")
                        
                elif user_input.startswith('review'):
                    parts = user_input.split(' ', 1)
                    if len(parts) > 1:
                        # User provided a new question to ask and review
                        question = parts[1].strip()
                        pipeline.ask_question(question, review=True)
                    else:
                        # User wants to review the last question's retrieval results
                        if pipeline.last_message_id:
                            pipeline.review_retrieval(pipeline.last_message_id)
                        else:
                            print("No previous question to review. Ask a question first or use 'review <question>'")
                    
                else:
                    print("Unknown command. Available commands:")
                    print("  - ask <question>")
                    print("  - review [question]")
                    print("  - quit")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    elif args.command == "ask":
        pipeline.setup_pipeline()
        pipeline.ask_question(args.prompt, review=args.review)
    elif args.command == "review":
        pipeline.setup_pipeline()
        pipeline.ask_question(args.prompt, review=True)
    elif args.command == "cleanup":
        pipeline.cleanup()
    else:
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()

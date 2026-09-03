"""llmcli - a small toolchain around any OpenAI-compatible LLM endpoint.

Modules
    llmcli.tools     workspace config + all communication with the models
    llmcli.corpus    documents: read, chunk, embed, retrieve, ask
    llmcli.chat      interactive chat REPL
    llmcli.dupcheck  duplicate / near-duplicate code detection

tools owns every request that leaves the machine; corpus and dupcheck are
peers that hand it data to process - one works with documents, the other
with source code (which it parses into an AST first).
"""

__version__ = "2.0.0"

"""An agentic RAG system over SEC 10-K filings.

    config      settings: models, chunking, k, index name
    embed       Cloudflare Workers AI embeddings
    decide      the decision model: scope() and diagnose()
    retrieval   the five retrieval strategies
    agent       the router, the diagnosis, the graph, and the deployed agent
    ladder      the fixed-escalation baseline agent
    evaluate    retrieval metrics, the LLM judge, and agent comparison
    util        rate-limit retry and reply text extraction
"""

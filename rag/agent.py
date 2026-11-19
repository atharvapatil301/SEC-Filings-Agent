"""The diagnosis agent: a decision model routes and judges, a chat model writes.

    route (decision) -+-> lookup (figure questions, the financials file) -> answer
                      +-> retrieve -> diagnose (decision) -> decide
                                                             |-- sufficient / absent -> answer (chat)
                                                             |-- wrong_words        -> stronger search, back to retrieve
                                                             |-- incomplete         -> plan (chat), execute, back to diagnose
"""
import json
import operator
import os
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from . import config, decide as decision, retrieval
from .util import retry, text_of

MAX_ATTEMPTS = 3

# What each kind of question starts with. Measured in the retrieval
# comparison: the filter wins on specific questions and loses on comparative
# ones, where hybrid wins.
START_WITH = {"specific": "filter", "comparative": "hybrid+rrf", "figure": "filter"}
DEFAULT = "filter+rrf"            # when the router is not sure
SURE = 0.6                        # below this, use DEFAULT

# wrong_words means the search was aimed right and missed, so search harder.
STRONGER = {"filter": "filter+rrf", "hybrid+rrf": "filter+rrf+rerank",
            "filter+rrf": "filter+rrf+rerank"}

# absent is a judgment, and the model can be unsure of it. Below this I
# give it one more search before taking its word.
ABSENT_SURE = 0.8


# ---------------------------------------------------------------- the tools

def financial_fact(ticker: str, concept: str) -> str:
    """Look up a reported figure. ticker is NVDA, AMD or MU. concept is an
    XBRL name such as Revenues or ResearchAndDevelopmentExpense."""
    path = os.path.join(config.FINANCIALS, f"{ticker.upper()}_financials.json")
    if not os.path.exists(path):
        return f"No financials for {ticker}. Use NVDA, AMD or MU."
    facts = json.load(open(path))["facts"]
    if concept not in facts:
        return (f"{ticker} has no concept named {concept}. Available: "
                + ", ".join(sorted(facts)[:25]))
    rows = sorted(facts[concept].items())
    return f"{ticker} {concept}:\n" + "\n".join(
        f"  period ending {d}: {v:,}" for d, v in rows)


def format_passages(hits):
    return "\n\n".join(f"[{source}] {text}" for source, text in hits)


# ---------------------------------------------------------------- the state

class RetrievalState(TypedDict, total=False):
    question: str
    scope: str                 # specific, comparative, figure
    strategy: str              # which retrieval to run next
    documents: Annotated[list, operator.add]
    verdict: str               # sufficient, absent, wrong_words, incomplete
    probabilities: dict
    steps: list
    attempts: int
    chat_calls: Annotated[int, operator.add]
    decision_calls: Annotated[int, operator.add]
    answer: str


# ---------------------------------------------------------------- the nodes

def route(state: RetrievalState) -> dict:
    """One decision-model call. Which kind of question, and so which search.

    decide.scope returns a choice and its probabilities. The strategy comes
    from START_WITH for that choice, or DEFAULT when the top probability is
    below SURE. Counts one decision call.
    """
    scope, probs  = decision.scope(state["question"])
    strategy=""
    if probs[scope] >= SURE:
        strategy = START_WITH[scope]
    else:
        strategy=DEFAULT 

    return {
        "scope": scope,
        "strategy": strategy,
        "decision_calls": 1
    }
    

def after_route(state: RetrievalState) -> Literal["lookup", "retrieve"]:
    """Figure questions do not go to retrieval at all."""
    return "lookup" if state["scope"] == "figure" else "retrieve"

class Fact(BaseModel):
    """Which figure a question asks for."""
    ticker: Literal["NVDA", "AMD", "MU"]
    concept: Literal["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                     "ResearchAndDevelopmentExpense", "GrossProfit", "NetIncomeLoss",
                     "OperatingIncomeLoss"] = Field(
        description="the XBRL concept. Revenue questions: Revenues, or "
                    "RevenueFromContractWithCustomerExcludingAssessedTax for AMD. "
                    "R&D spend: ResearchAndDevelopmentExpense.")


FACT_PROMPT = """Which reported figure does this question ask for?

  research and development spend  -> ResearchAndDevelopmentExpense
  revenue, sales, top line         -> Revenues (AMD reports it as
                                      RevenueFromContractWithCustomerExcludingAssessedTax)
  gross profit, gross margin       -> GrossProfit
  net income, profit, earnings     -> NetIncomeLoss
  operating income                 -> OperatingIncomeLoss

ticker is the company's symbol: NVIDIA -> NVDA, AMD -> AMD, Micron -> MU.

Example: "What did Micron spend on R&D in fiscal 2025?"
         -> ticker=MU, concept=ResearchAndDevelopmentExpense

Question: {question}"""


def lookup(state: RetrievalState) -> dict:
    """A number is in the financials file, not in the prose. One chat call
    to name the figure, then a plain dictionary lookup.

    The chat model names a Fact, financial_fact reads it, and the table is
    returned as the one document with the verdict already sufficient, so the
    graph goes straight to answer. Counts one attempt and one chat call.
    """
    namer = config.chat().with_structured_output(Fact)
    fact = retry(namer.invoke, FACT_PROMPT.format(question=state["question"]))
    table = financial_fact(fact.ticker, fact.concept)
    return {"documents": [(f"{fact.ticker}_financials.json", table)],
            "verdict": "sufficient", "attempts": 1, "chat_calls": 1}

def retrieve(state: RetrievalState) -> dict:
    search = retrieval.STRATEGIES[state["strategy"]]
    hits = search(state["question"], config.K)
    seen = set(state.get("documents") or [])
    new = [h for h in hits if h not in seen]
    chat = {"filter": 1, "hybrid+rrf": 1, "filter+rrf": 2, "filter+rrf+rerank": 2}
    return {"documents": new, "attempts": state.get("attempts", 0) + 1,
            "chat_calls": chat[state["strategy"]]}


def diagnose(state: RetrievalState) -> dict:
    """One decision-model call. Enough, or why not.

    decide.diagnose returns the verdict and its probabilities. absent is a
    judgment, and the model can be unsure of it: an absent below ABSENT_SURE,
    with attempts left, is treated as wrong_words so the graph searches once
    more. Counts one decision call.
    """
    verdict, probs = decision.diagnose(state["question"], state["documents"])
    
    if (verdict == "absent" and probs["absent"] < ABSENT_SURE
            and state.get("attempts", 0) < MAX_ATTEMPTS):
        verdict = "wrong_words"
    return {"verdict": verdict, "probabilities": probs, "decision_calls": 1}

def decide(state: RetrievalState) -> Literal["answer", "stronger", "plan"]:
    """Every verdict has one place to go. sufficient and absent both stop.
    incomplete needs a plan. wrong_words needs a stronger search, unless there
    is none left. And the attempt cap stops everything."""
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "answer"
    if state["verdict"] == "incomplete":
        return "plan"
    if state["verdict"] == "wrong_words" and state["strategy"] in STRONGER:
        return "stronger"
    return "answer"

def stronger(state: RetrievalState) -> dict:
    """Same question, the next search up. STRONGER says which.
    """
    return {"strategy": STRONGER[state["strategy"]]}

class Plan(BaseModel):
    """The separate questions a comparative question breaks into."""
    steps: list[str] = Field(description="Each names one company, one year if "
                                         "relevant, and one thing to find")


PLAN_PROMPT = """A question about SEC 10-K filings needs more than one search:

{question}

Passages found so far cover: {covered}

Write the separate questions still needed. Each must name exactly one company
and ask for one thing, so that a filtered search can find it. Fewest possible."""


def plan(state: RetrievalState) -> dict:
    """One chat-model call. Break the question into per-company searches.

    PLAN_PROMPT wants to know what the passages so far already cover, so the
    planner does not ask for it again. The sources of the documents in state
    are enough for that.
    """
    planner = config.chat().with_structured_output(Plan)
    sources = sorted({source for source, _ in state.get("documents") or []})
    covered = ", ".join(sources) or "nothing"
    out = retry(planner.invoke, PLAN_PROMPT.format(question=state["question"], covered=covered))
    return {"steps": out.steps, "chat_calls": 1}

def execute(state: RetrievalState) -> dict:
    """One filtered search per step. The filter extractor is a chat call each.

    Returns only passages not already in the state. documents has an add
    reducer, so the graph does the appending.
    """
    seen = set(state.get("documents") or [])
    new = []
    for step in state["steps"]:
        for hit in retrieval.with_filter(step, config.K):
            if hit not in seen:
                seen.add(hit)
                new.append(hit)
    return {"documents": new, "attempts": state.get("attempts", 0) + 1,
            "chat_calls": len(state["steps"])}

ANSWER_PROMPT = """Answer the question using only these passages from SEC
filings. Cite the filing each fact comes from, by its filename in brackets.
If the passages do not cover the question, say so in one sentence.

Question: {question}

Passages:
{passages}"""


def answer(state: RetrievalState) -> dict:
    """One chat-model call. The only place prose gets written."""
    if state.get("verdict") == "absent":
        return {"answer": "The filings in this corpus do not cover that.", "chat_calls": 0}
    out = retry(config.chat().invoke, ANSWER_PROMPT.format(
        question=state["question"], passages=format_passages(state["documents"])))
    return {"answer": text_of(out), "chat_calls": 1}


def build_graph():
    """The retrieval graph. decide and after_route return Literals, which
    declare their conditional edges."""
    from langgraph.graph import StateGraph, START, END

    agent = StateGraph(RetrievalState)
    agent.add_node("route", route)
    agent.add_node("retrieve", retrieve)
    agent.add_node("diagnose", diagnose)
    agent.add_node("plan", plan)
    agent.add_node("stronger", stronger)
    agent.add_node("execute", execute)
    agent.add_node("lookup", lookup)
    agent.add_node("answer", answer)


    agent.add_edge(START, "route")
    agent.add_conditional_edges("route", after_route)
    agent.add_edge("lookup", "answer")
    agent.add_edge("answer", END)

    agent.add_edge("retrieve", "diagnose")
    agent.add_conditional_edges("diagnose",decide)
    agent.add_edge("plan", "execute")

    agent.add_edge("stronger", "retrieve")
    agent.add_edge("execute", "diagnose")

    

    return agent.compile()

def run(question):
    """The whole final state, for tracing and for the cost accounting in evaluate."""
    return build_graph().invoke({"question": question, "attempts": 0,
                                 "chat_calls": 0, "decision_calls": 0})


def ask(question, verbose=False):
    state = run(question)
    if verbose:
        print(f"  {state['scope']} -> {state['strategy']}, verdict {state['verdict']}, "
              f"{state['attempts']} attempt(s), {state['chat_calls']} chat + "
              f"{state['decision_calls']} decision calls")
    return state["answer"]


# ---------------------------------------------------------------- the agent
#
# The graph above is the retrieval. The agent is what app.py serves: the same
# create_agent with two tools, the graph behind search_filings and
# the financials file behind financial_fact, and a checkpointer so the chat
# remembers the conversation. agentui.py streams it, tool calls and all.

def search_filings(question: str) -> str:
    """Answer a question from the text of the 10-K filings of NVIDIA, AMD and
    Micron: business, risk factors, management's discussion. Returns an answer
    with the filing each fact came from in brackets. Use financial_fact instead
    for a reported number such as revenue or R&D spend."""
    return run(question)["answer"]


SYSTEM_PROMPT = """You answer questions about NVIDIA, AMD and Micron from their
10-K filings. Use search_filings for anything the filings say in prose, and
financial_fact for a reported number. Relay the tool's answer, with its
citations, and do not add facts of your own. If the tools find nothing, say so."""


def build_agent():
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import MemorySaver

    return create_agent(config.chat(), tools=[search_filings, financial_fact],
                        system_prompt=SYSTEM_PROMPT, checkpointer=MemorySaver())

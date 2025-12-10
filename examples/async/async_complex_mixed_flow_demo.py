"""
Async Investment Analysis Demo (Parallelism Showcase)

This scenario demonstrates the power of Async Agents in a time-critical domain: Financial Analysis.
The goal is to analyze a target company (e.g., "NVIDIA") from multiple angles simultaneously.

Architecture:
    [HIERARCHICAL ROOT]
             │
        InvestmentDirector (Root)
             │
             ├─── [TEAM: Research] 
             │    HeadOfResearch (Supervisor)
             │         ├─ MacroAnalyst   (Checks global economy)
             │         ├─ TechAnalyst    (Checks technology/product)
             │         └─ MarketAnalyst  (Checks competitors/trends)
             │
             └─── [PIPELINE: Sequential Risk Check]
                       │
                       ├─ Step 1: ComplianceOfficer (Checks legal restrictions)
                       │
                       └─ Step 2: RiskManager (Calculates portfolio exposure)

Key Feature: The 'HeadOfResearch' is instructed to launch all 3 analysts at once.
In a sync system, this would take (Time A + Time B + Time C).
In this async system, it takes MAX(Time A, Time B, Time C).
"""

import asyncio
import os
import sys
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agentify.core import BaseAgent, AgentConfig
from agentify.memory import MemoryService
from agentify.memory.stores.in_memory_store import InMemoryStore
from agentify.multi_agent import Team, Pipeline, HierarchicalTeam


def create_investment_system():
    """Creates the investment analysis multi-agent system."""

    store = InMemoryStore()
    memory_service = MemoryService(store=store, log_enabled=True)

    # --- 1. Research Team (The Parallel Engine) ---
    
    macro_analyst = BaseAgent(
        config=AgentConfig(
            name="MacroAnalyst",
            system_prompt="You are a Macroeconomic Analyst. Analyze the current global economic environment (inflation, rates, geopolitics) relevant to the tech sector. Be concise.",
            provider="deepseek",
            model_name="deepseek-reasoner", # Thinking model
            reasoning_effort="high",
            temperature=0.3
        ),
        memory=memory_service
    )

    tech_analyst = BaseAgent(
        config=AgentConfig(
            name="TechAnalyst",
            system_prompt="You are a Technology Analyst. Evaluate the target company's product stack, innovation moat, and technical debt. Be concise.",
            provider="deepseek",
            model_name="deepseek-reasoner",
            reasoning_effort="high",
            temperature=0.3
        ),
        memory=memory_service
    )

    market_analyst = BaseAgent(
        config=AgentConfig(
            name="MarketAnalyst",
            system_prompt="You are a Market Sentiment Analyst. Analyze competitor moves and market sentiment trends for the target. Be concise.",
            provider="deepseek",
            model_name="deepseek-reasoner",
            reasoning_effort="high",
            temperature=0.3
        ),
        memory=memory_service
    )

    head_of_research = BaseAgent(
        config=AgentConfig(
            name="HeadOfResearch",
            system_prompt=(
                "You are the Head of Research. Your goal is to produce a comprehensive 360-degree report on a target company. "
                "You manage three analysts: MacroAnalyst, TechAnalyst, and MarketAnalyst. "
                "CRITICAL INSTRUCTION: To speed up the process, you MUST call all three analysts SIMULTANEOUSLY in the same turn. "
                "Do not wait for one to finish before calling the others. "
                "Once you have all reports, synthesize them into a single summary."
            ),
            provider="deepseek",
            model_name="deepseek-reasoner",
            reasoning_effort="high",
            temperature=0.2,
            max_tool_iter=5
        ),
        memory=memory_service
    )

    research_team = Team(
        agents=[head_of_research, macro_analyst, tech_analyst, market_analyst],
        supervisor=head_of_research
    )
    research_team.name = "ResearchTeam"
    research_team.description = "Deep dive research team. Use this to get a comprehensive analysis of the company."

    # --- 2. Risk Pipeline (The Sequential Check) ---

    compliance_officer = BaseAgent(
        config=AgentConfig(
            name="ComplianceOfficer",
            system_prompt="You are a Compliance Officer. Check the proposed trade/analysis for any regulatory issues (SEC restrictions, insider lists, etc). If clean, say 'APPROVED'.",
            provider="deepseek",
            model_name="deepseek-reasoner",
            reasoning_effort="high",
            temperature=0.1
        ),
        memory=memory_service
    )

    risk_manager = BaseAgent(
        config=AgentConfig(
            name="RiskManager",
            system_prompt="You are a Risk Manager. Receive the compliance status and the research summary. Calculate portfolio exposure and recommend position sizing (Small/Medium/Large).",
            provider="deepseek",
            model_name="deepseek-reasoner",
            reasoning_effort="high",
            temperature=0.2
        ),
        memory=memory_service
    )

    risk_pipeline = Pipeline(steps=[compliance_officer, risk_manager])
    risk_pipeline.name = "RiskPipeline"
    risk_pipeline.description = "Risk and Compliance check. Must be run AFTER research."

    # --- 3. Root ---

    director = BaseAgent(
        config=AgentConfig(
            name="InvestmentDirector",
            system_prompt=(
                "You are the Investment Director. You decide on stock allocations. "
                "1. First, ask the ResearchTeam for a full analysis of the target. "
                "2. Then, send their findings to the RiskPipeline to get approval and sizing. "
                "3. Finally, issue a final BUY/SELL/HOLD recommendation based on everything."
            ),
            provider="deepseek",
            model_name="deepseek-chat",
            temperature=0.4
        ),
        memory=memory_service
    )

    hierarchy = {director: [research_team, risk_pipeline]}
    return HierarchicalTeam(root=director, hierarchy=hierarchy)


async def run_investment_demo():
    print("=" * 80)
    print("ASYNC INVESTMENT ANALYSIS DEMO (High Parallelism)")
    print("=" * 80)
    print("Scenario: Analyzing 'Tesla (TSLA)'")
    print("Structure: Director -> [ResearchTeam (3 Parallel Analysts)] -> [RiskPipeline (Sequential)]")
    print("-" * 80)

    system = create_investment_system()
    
    user_request = "Should we invest in Tesla (TSLA) right now? I need a deep analysis."

    start_time = time.time()
    
    result = await system.arun(
        user_input=user_request,
        session_id="inv_demo_001",
        user_id="trader_vip"
    )

    end_time = time.time()
    duration = end_time - start_time

    print("=" * 80)
    print("FINAL RECOMMENDATION:")
    print("-" * 80)
    print(result)
    print("=" * 80)
    print(f"Total Execution Time: {duration:.2f} seconds")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(run_investment_demo())

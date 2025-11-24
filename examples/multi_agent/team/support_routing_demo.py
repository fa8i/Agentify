import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agentify.core import BaseAgent, AgentConfig
from agentify.memory import MemoryService
from agentify.memory.stores.in_memory_store import InMemoryStore
from agentify.multi_agent import Team
from agentify.extensions.tools import get_current_time_tool, calculate_expression_tool

load_dotenv()


def create_support_team():
    memory = MemoryService(store=InMemoryStore())

    # --- Workers ---

    billing_config = AgentConfig(
        name="BillingExpert",
        system_prompt=(
            "You are a specialist in SaaS B2B billing. "
            "You explain clearly topics of invoices, taxes, discounts, and billing cycles. "
            "You are very concrete and get to the point."
        ),
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.0,
    )
    billing_agent = BaseAgent(
        config=billing_config,
        memory=memory,
    )

    tech_config = AgentConfig(
        name="TechExpert",
        system_prompt=(
            "You are a support engineer. "
            "You help with API errors, integration problems, and performance. "
            "You can use calculation and current time tools to reason better."
        ),
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.1,
    )
    tech_agent = BaseAgent(
        config=tech_config,
        memory=memory,
        tools=[get_current_time_tool, calculate_expression_tool],
    )

    success_config = AgentConfig(
        name="SuccessCoach",
        system_prompt=(
            "You are a Customer Success Manager. "
            "You give product usage recommendations, best practices "
            "and help organize adoption in the team."
        ),
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.5,
    )
    success_agent = BaseAgent(
        config=success_config,
        memory=memory,
    )

    # --- Supervisor / Router ---

    router_config = AgentConfig(
        name="Router",
        system_prompt=(
            "You are the support router for a SaaS company."
            "- Classify each query as: billing, technical, or customer success."
            "- Use the available tools to delegate to the appropriate specialist "
            "(BillingExpert, TechExpert, or SuccessCoach)."
            "- Do not explain to the user that you are using internal agents."
            "- After receiving the specialist's response, summarize and adapt the tone to the user "
            "in 3-6 clear sentences."
            "- If the query is extremely trivial, you can answer directly without delegating."
        ),
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.2,
    )

    router_agent = BaseAgent(
        config=router_config,
        memory=memory,
    )

    # Team con topología plana: Router -> [Billing, Tech, Success]
    team = Team(
        agents=[router_agent, billing_agent, tech_agent, success_agent],
        supervisor=router_agent,
    )
    return team


def main():
    print("Initializing support team (router + specialists)...")
    team = create_support_team()

    session_id = "support_routing_demo"

    queries = [
        "I don't understand why you charged me almost double this month compared to last.",
        "I'm getting 500 errors when calling your API from production every few hours.",
        "We want to extend the use of the tool to the sales team; how would you organize that?",
    ]

    for i, query in enumerate(queries, start=1):
        print(f"\n=== Ticket {i} ===")
        print(f"User: {query}\n")
        response = team.run(
            user_input=query,
            session_id=session_id,
            user_id="customer_123",
        )
        print("Team response:\n")
        print(response)


if __name__ == "__main__":
    main()

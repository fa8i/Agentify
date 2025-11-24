"""
Complex Mixed Multi-Agent Flow Demo

This example demonstrates the composability of Agentify's multi-agent patterns
by combining all three orchestration patterns in a single flow:

Architecture:
    [HIERARCHICAL ROOT]
             │
        ProductOwner (Root)
             │
             ├─── [TEAM 1] DesignTeam (supervisor: Architect)
             │         ├─ UXDesigner
             │         └─ TechWriter
             │
             └─── [SEQUENTIAL PIPELINE]
                       │
                       ├─ Step 1: [TEAM 2] DevTeam (supervisor: TechLead)
                       │              ├─ BackendDev
                       │              └─ FrontendDev
                       │
                       └─ Step 2: QAEngineer (single agent)

Flow:
    User Request → ProductOwner → DesignTeam → Pipeline(DevTeam → QA) → Final Output

This tests:
    1. Deep nesting (4 levels of delegation)
    2. Pattern composition (Hierarchical + Team + Sequential)
    3. Memory addressing across different organizational structures
    4. Practical multi-agent coordination
"""

import os
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agentify.core import BaseAgent, AgentConfig
from agentify.memory.service import MemoryService
from agentify.memory.stores.in_memory_store import InMemoryStore
from agentify.multi_agent import Team
from agentify.multi_agent.pipeline import SequentialPipeline
from agentify.multi_agent.hierarchical import HierarchicalTeam


def create_complex_system():
    """Creates the complex multi-agent system with nested patterns."""

    # Shared memory service
    memory = MemoryService(store=InMemoryStore())

    # ======================
    # Level 3: Specialists
    # ======================

    # Design Team Members
    ux_designer = BaseAgent(
        config=AgentConfig(
            name="UXDesigner",
            system_prompt=(
                "You are a UX Designer. You create user experience specifications. "
                "Focus on user flows, wireframes concepts, and interaction patterns. "
                "Be brief and practical."
            ),
            provider="deepseek",
            model_name="deepseek-chat",
            temperature=0.6,
        ),
        memory=memory,
    )

    tech_writer = BaseAgent(
        config=AgentConfig(
            name="TechWriter",
            system_prompt=(
                "You are a Technical Writer. You document API specifications and user guides. "
                "Create clear, structured documentation with examples. "
                "Be concise and technical."
            ),
            provider="deepseek",
            model_name="deepseek-chat",
            temperature=0.3,
        ),
        memory=memory,
    )

    # Dev Team Members
    backend_dev = BaseAgent(
        config=AgentConfig(
            name="BackendDev",
            system_prompt=(
                "You are a Backend Developer specializing in Python/FastAPI. "
                "You write clean, RESTful API code with proper error handling. "
                "Provide code snippets with brief explanations."
            ),
            provider="deepseek",
            model_name="deepseek-chat",
            temperature=0.2,
        ),
        memory=memory,
    )

    frontend_dev = BaseAgent(
        config=AgentConfig(
            name="FrontendDev",
            system_prompt=(
                "You are a Frontend Developer working with vanilla JavaScript. "
                "You create simple, functional UI code that consumes REST APIs. "
                "Focus on practical implementations."
            ),
            provider="deepseek",
            model_name="deepseek-chat",
            temperature=0.2,
        ),
        memory=memory,
    )

    # QA Engineer (standalone)
    qa_engineer = BaseAgent(
        config=AgentConfig(
            name="QAEngineer",
            system_prompt=(
                "You are a QA Engineer. You review code and designs for bugs, edge cases, "
                "and potential issues. Provide a brief quality report with: "
                "1) Issues found, 2) Severity assessment, 3) Recommendations. "
                "Be critical but constructive."
            ),
            provider="deepseek",
            model_name="deepseek-chat",
            temperature=0.4,
        ),
        memory=memory,
    )

    # ======================
    # Level 2: Team Leads
    # ======================

    # Architect (supervises Design Team)
    architect = BaseAgent(
        config=AgentConfig(
            name="Architect",
            system_prompt=(
                "You are a Software Architect. You coordinate the UX Designer and Tech Writer "
                "to create comprehensive design specifications. "
                "First get UX flows from UXDesigner, then API documentation from TechWriter. "
                "Synthesize their outputs into a cohesive design document. "
                "Don't reveal that you're delegating; present a unified result."
            ),
            provider="deepseek",
            model_name="deepseek-chat",
            temperature=0.3,
            max_tool_iter=5,
        ),
        memory=memory,
    )

    # Tech Lead (supervises Dev Team)
    tech_lead = BaseAgent(
        config=AgentConfig(
            name="TechLead",
            system_prompt=(
                "You are a Tech Lead managing Backend and Frontend developers. "
                "Delegate backend tasks to BackendDev and frontend tasks to FrontendDev. "
                "Coordinate their work and combine it into a complete implementation. "
                "Present the final code as a unified delivery."
            ),
            provider="deepseek",
            model_name="deepseek-chat",
            temperature=0.3,
            max_tool_iter=5,
        ),
        memory=memory,
    )

    # ======================
    # Level 1: Teams
    # ======================

    design_team = Team(
        agents=[architect, ux_designer, tech_writer],
        supervisor=architect,
    )

    dev_team = Team(
        agents=[tech_lead, backend_dev, frontend_dev],
        supervisor=tech_lead,
    )

    # ======================
    # Level 0.5: Pipeline
    # ======================

    # Sequential: Dev → QA
    dev_pipeline = SequentialPipeline(steps=[dev_team, qa_engineer])

    # ======================
    # Level 0: Root
    # ======================

    product_owner = BaseAgent(
        config=AgentConfig(
            name="ProductOwner",
            system_prompt=(
                "You are a Product Owner coordinating a complete software development process. "
                "When you receive a feature request: "
                "1. First, delegate to the Architect to get design specifications "
                "2. Then, delegate to TechLead to implement and test the feature "
                "Important: The TechLead manages a complete dev pipeline including QA. "
                "Finally, synthesize everything into a coherent final delivery report. "
                "Present results as if the entire team worked seamlessly together."
            ),
            provider="deepseek",
            model_name="deepseek-chat",
            temperature=0.4,
            max_tool_iter=8,
        ),
        memory=memory,
    )

    # ======================
    # Hierarchical Assembly
    # ======================

    # We can directly nest Team and Pipeline objects
    # ProductOwner → [DesignTeam, DevPipeline]

    # Add names/descriptions to the flow objects so FlowTool can use them
    design_team.name = "DesignTeam"
    design_team.description = (
        "Delegate to the Design Team (Architect, UX, Writer) to create specifications."
    )

    dev_pipeline.name = "DevPipeline"
    dev_pipeline.description = "Delegate to the Development Pipeline (DevTeam + QA) to implement and verify code."

    hierarchy = {product_owner: [design_team, dev_pipeline]}

    # Create the top-level hierarchical team
    full_system = HierarchicalTeam(root=product_owner, hierarchy=hierarchy)

    return full_system


def run_complex_flow():
    """Executes the complex multi-agent workflow."""

    print("=" * 80)
    print("COMPLEX MULTI-AGENT FLOW DEMO (FIXED)")
    print("=" * 80)
    print("\nInitializing system with 7 agents in nested structure...")
    print("  - ProductOwner (root)")
    print("  - DesignTeam: Architect → [UXDesigner, TechWriter]")
    print(
        "  - DevPipeline: [DevTeam: TechLead → [BackendDev, FrontendDev]] → QAEngineer"
    )
    print()

    # Now returns a HierarchicalTeam instance
    team = create_complex_system()

    session_id = "complex_flow_demo_fixed_001"
    user_id = "stakeholder_001"

    request = (
        "Create a simple TODO list API with these requirements: "
        "- REST endpoints: GET /todos, POST /todos, DELETE /todos/:id "
        "- Simple in-memory storage "
        "- Include a minimal HTML interface to test it "
        "- Keep it under 200 lines total"
    )

    print("=" * 80)
    print("USER REQUEST:")
    print("-" * 80)
    print(request)
    print()

    # Run the entire flow from the root
    print("Running full hierarchical flow...")
    result = team.run(
        user_input=request,
        session_id=session_id,
        user_id=user_id,
    )

    print("=" * 80)
    print("FINAL OUTPUT:")
    print("-" * 80)
    print(result)


if __name__ == "__main__":
    run_complex_flow()

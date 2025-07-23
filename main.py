import asyncio
import os

import chainlit as cl

from agents import Agent, Runner, OpenAIChatCompletionsModel
from agents.run import RunConfig
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI
from typing import cast

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("Your Api key is not set please check the .env file.")

def set_config():
    client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    model = OpenAIChatCompletionsModel(
        model="gemini-2.5-flash",
        openai_client=client
    )
    config = RunConfig(
        model=model,
        model_provider=client,
        tracing_disabled=True
    )
    Scalp_Analyzer=Agent(
        name="ScalpAnalyzer Agent",
        instructions="Analyze scalp condition based on input symptoms (e.g., dryness, oiliness, dandruff).",
        handoff_description="Used when a scalp-related diagnosis is needed before recommending treatments.",
        model=model
    )
    Product_Recommender = Agent(
        name="ProductRecommender Agent",
        instructions="Suggest personalized hair products based on scalp type, hair goals, and budget.",
        handoff_description="Activated after scalp diagnosis or when the user directly asks for product advice.",
        model=model
    )
    Treatment_planner = Agent(
        name="TreatmentPlanner Agent",
        instructions="Create a step-by-step weekly hair care routine based on user needs (e.g., hair fall, volume, growth).",
        handoff_description="Triggered when a user asks for a routine, recovery plan, or ongoing treatment.",
        model=model
    )
    Diet_Advisor = Agent(
        name="DietAdvisor Agent",
        instructions="Recommend foods or supplements that support healthy hair from within.",
        handoff_description="Used when user mentions nutritional concerns or wants internal solutions.",
        model=model
    )
    HairLoss_Specialist =Agent(
        name="HairLossSpecialist Agent",
        instructions="Analyze hair fall patterns and recommend medical or advanced care.",
        handoff_description="Activated when user talks about hair thinning, patchy loss, or genetic hair loss.",
        model=model
    )
    Scalp_Analyzer_tool = Scalp_Analyzer.as_tool(tool_name="ScalpAnalyzer",tool_description="Analyzes the user's scalp condition based on reported symptoms such as dryness, oiliness, flakiness, itching, or redness. It identifies potential scalp disorders (e.g., dandruff, seborrheic dermatitis, psoriasis) and prepares diagnostic context for treatment or product recommendations.")
    Product_Recommender_tool = Product_Recommender.as_tool(tool_name="Product_Recomemnder",tool_description="Suggests personalized hair care products (shampoos, conditioners, serums, oils) based on the user’s scalp type, hair issues, preferences, and budget. Ensures compatibility with diagnoses provided by other agents like ScalpAnalyzer or HairLossSpecialist.")
    Treatment_planner_tool = Treatment_planner.as_tool(tool_name="Treatment_Planner",tool_description="Creates a detailed and personalized weekly or daily hair care routine. This includes when and how to apply products, washing schedules, scalp treatments, and maintenance tips. It considers diagnostic data from other agents and user preferences (e.g., minimal steps or intensive care).")
    Diet_Advisor_tool = Diet_Advisor.as_tool(tool_name="Diet_Advisor",tool_description="Recommends diet improvements and supplements (vitamins, minerals, hydration tips) to support internal hair health. Tailors suggestions based on issues like hair fall, slow growth, or scalp inflammation, bridging nutrition with hair care.")
    HairLoss_Specialist_tool = HairLoss_Specialist.as_tool(tool_name="Hairloss_Specialist",tool_description="Assesses hair thinning and hair fall severity, patterns, and potential causes (genetics, hormonal imbalance, stress, etc.). Recommends appropriate action such as medical consultation, product use, or advanced care routines. Prepares input for TreatmentPlanner if needed.")

    instruction = """
    > You are the **HairCare Specialist**, a master controller agent responsible for managing a team of specialized hair care agents. Your goal is to understand the user's concerns and intelligently delegate tasks to the appropriate sub-agents.
    >
    > 🎯 **Your Responsibilities:**
    >
    > 1. Interpret user input related to hair issues, scalp condition, product needs, or routines.
    > 2. Decide **which sub-agent(s)** are best suited for the task and **handoff** control to them.
    > 3. Coordinate the flow between multiple agents when more than one is needed.
    > 4. Ensure user gets a **complete, personalized solution** by gathering results from agents and combining them into a single, helpful response.
    >
    > 📦 **Available Sub-Agents and Their Roles:**
    >
    > * `ScalpAnalyzer`: Diagnoses scalp conditions.
    > * `ProductRecommender`: Suggests hair care products.
    > * `TreatmentPlanner`: Creates personalized hair care routines.
    > * `DietAdvisor`: Recommends diet/supplements for healthy hair.
    > * `HairLossSpecialist`: Handles advanced hair loss concerns.
    >
    > 🔄 **Interaction Strategy:**
    >
    > * Use the **minimum number of agents** required to solve the user's problem fully.
    > * Provide a **summary or report** to the user combining all relevant outputs."""

    Orchestrator_agent = Agent(
        name="HairCare Specialist Agent",
        instructions=instruction,
        tools=[Scalp_Analyzer_tool,Product_Recommender_tool,Treatment_planner_tool,Diet_Advisor_tool,HairLoss_Specialist_tool],
        model=model
    )
    return Orchestrator_agent,config

@cl.on_chat_start
async def chat_start():
    Orchestrator_agent,config = set_config()
    cl.user_session.set("chat_history",[])
    cl.user_session.set("config",config)
    cl.user_session.set("Orchestrator_agent",Orchestrator_agent)
    await cl.Message(
        content=' HairCareOS – AI-Powered Multi-Agent System for Scalp & Hair Health'
    ).send()

@cl.on_message    
async def main(message : cl.Message):

    msg = cl.Message(content='Thinking ...')
    await msg.send()

    Orchestrator_agent :Agent = cast(Agent,cl.user_session.get("Orchestrator_agent"))
    config :RunConfig = cast(RunConfig,cl.user_session.get("config"))

    history = cl.user_session.get("chat_history") or []
    history.append({
        "role":"user",
        "content":message.content
    })
    
    try:

        print("\nCalling_Agent_with_Context\n",history,"\n")

        response = await Runner.run(
            starting_agent=Orchestrator_agent,
            input=history,
            run_config=config
        )

        response_content = response.final_output

        msg.content = response_content
        await msg.update()

        history.append({
            "role":"assistant",
            "content": response_content
        })

        cl.user_session.set("chat_history",history)

        print(f'User : {message.content}')
        print(f'Assistant : {response_content}')
    
    except Exception as e:

        msg.content = f'Error : {str(e)}'
        await msg.update()
        print(f'Error : {str(e)}')
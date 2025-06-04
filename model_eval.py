# Response evauluator
#https://docs.smith.langchain.com/evaluation/concepts#evaluating-an-agents-final-response
from typing import Literal
from main import graph
import json
import asyncio
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.types import Command, interrupt
from typing_extensions import Annotated, TypedDict
from langsmith import Client
from main import set_env

set_env()

client = Client()

examples = [
    {
        "inputs": {
            "question": "What is a Google Merchant?",
        },
        "outputs": {
            "response": "Google Merchant Center is a centralized dashboard"
            "that allows you to organize your e-commerce products to appear"
            "in Google shopping searches, allowing potential customers to browse your products via Google."
        }
    },
    {
        "inputs": {
            "question": "Do I need to have a Google account?",
        },
        "outputs": {
            "response": "A standard Google account is a prerequisite."
            "If you have one, verify that you are logged in. If not, you can set one up."
        }
    },
    {
        "inputs": {
            "question": "Do I need to provide my personal data?",
        },
        "outputs": {
            "response": "Google will ask you to input details, including your name and contact information."
            "You'll also need to indicate whether your business is online, brick-and-mortar, or both."
        }
    },
    {
        "inputs": {
            "question": "Do I need to verify my website?"
        },
        "outputs": {
            "response": "Yes, to link your website to the Merchant Center account, Google requires you to verify ownership and claim your website URL. This can be done by adding an HTML tag or file to your site, or by using Google Analytics if it's already installed."
        }
    },
    {
        "inputs": {
            "question": "How do I handle sales tax when setting up my account?"
        },
        "outputs": {
            "response": "Google will suggest a tax setup based on your business location. You can allow Google to calculate the sales tax automatically or configure it manually. You’ll also be able to specify whether shipping and handling charges are taxable."
        }
    }
]

dataset_name = "SC Onboarding Agent"

if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(
        dataset_id=dataset.id,
        examples=examples
    )


# LLM-as-judge instructions
grader_instructions = """
You are a teacher grading a quiz.

You will be given a QUESTION, the GROUND TRUTH (correct) RESPONSE, and the STUDENT RESPONSE.

Grade the student responses based ONLY on their factual accuracy relative to the ground truth answer.
Ensure that the student response does not contain any conflicting statements.
It is OK if the student response contains more information than the ground truth response, as long as it is factually accurate relative to the ground truth response.

Your output MUST be a JSON object with these keys:
- reasoning: Explain your reasoning in a step-by-step manner.
- is_correct: true if the student's response is correct, otherwise false.

Example response:

{
  "reasoning": "The student's answer correctly covers all points without contradictions.",
  "is_correct": true
}

Respond ONLY with a valid JSON object — no extra text.
"""

# LLM-as-judge output schema
class Grade(TypedDict):
    """Compare the expected and actual answers and grade the actual answer."""
    reasoning: Annotated[str, ..., "Explain your reasoning for whether the actual response is correct or not."]
    is_correct: Annotated[bool, ..., "True if the student response is mostly or exactly correct, otherwise False."]


# Judge LLM
grader_llm = init_chat_model("llama3-70b-8192", temperature=0, model_provider="groq").with_structured_output(Grade, method="json_mode")


# Evaluator function
async def final_answer_correct(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """Evaluate if the final response is equivalent to reference response."""
    user = f"""QUESTION: {inputs['question']}
    GROUND TRUTH RESPONSE: {reference_outputs['response']}
    STUDENT RESPONSE: {outputs['response']}"""
    grade = await grader_llm.ainvoke([{"role": "system", "content": grader_instructions}, {"role": "user", "content": user}])
    return grade["is_correct"]


# Response eval
async def run_graph(inputs: dict) -> dict:
    result = await graph.ainvoke({
        "messages":
        [{ "role": "user", "content": inputs['question']}]},
        config = {"env": "test", "configurable": {"thread_id": "test_123"}})
    return {"response": result}


async def evaluate_model():
    return await client.aevaluate(
    run_graph,
    data=dataset_name,
    evaluators=[final_answer_correct],
    experiment_prefix="SC-Onboarding-Agent",
    num_repetitions=1,
    max_concurrency=4,
)


results = asyncio.run(evaluate_model())
results.to_pandas()
from pathlib import Path
import datetime

from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain_core.tools import Tool
from langchain.tools import PythonInterpreter
from wikipedia import summary


python_tool = PythonInterpreter()


class Agent:
    """
    base plan from beebot https://github.com/AutoPackAI/beebot
    [ Setup ]
        |
        | start
        v
    [ Oversight ]<------+
        |               |
        | decide        |
        v               |
    [ Deciding ]    [Planning]
        |               |
        | execute       |
        v               |
    [ Executing ]-------+
        |          plan
        | finish
        v
    [ Done ]

    """

    def __init__(self, system_prompt: str = "") -> None:
        self.system_prompt = system_prompt
        self.tools = [
            Tool(
                name="Time",
                func=self.get_current_time,
                description="Useful for when you need to know the current time. Get the current time.",
            ),
            Tool(
                name="Wikipedia",
                func=self.search_wikipedia,
                description="Useful for when you need to know information about a topic.",
            ),
            Tool(
                name="Python",
                func=python_tool.run,
                description="A Python interpreter. Use this to run Python code.",
            ),
        ]

    @staticmethod
    # Define a very simple tool function that returns the current time
    def get_current_time(*args, **kwargs):
        """Returns the current time in H:MM AM/PM format."""
        now = datetime.datetime.now()  # Get current time
        return now.strftime("%I:%M %p")  # Format time in H:MM AM/PM format

    @staticmethod
    def search_wikipedia(query: str):
        """Searches Wikipedia and returns the summary of the first result."""

        try:
            # Limit to two sentences for brevity
            return summary(query, sentences=5)
        except:
            return "I couldn't find any information on that."

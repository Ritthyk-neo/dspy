import dspy
from dotenv import dotenv_values, load_dotenv
from typing import Literal
import pydantic

load_dotenv()
config=dotenv_values("../.env")
endpoint = config['AZURE_OPENAI_ENDPOINT']
openai_api_key = config['AZURE_OPENAI_API_KEY']
openai_api_version = config['AZURE_OPENAI_API_VERSION']
deployment_id=config['DEPLOYMENT_ID']

lm= dspy.LM('azure/gpt-4o-mini',
            api_key=openai_api_key,
            api_base=endpoint,
            deployment_id=deployment_id,
            model_type="chat",
            api_version=openai_api_version)

dspy.configure(lm=lm)

#qa = 'question: str -> answer: str'
#mcq = 'question: str, choices: list[str] -> reasoning: str, answer: int'
#raqa - 'context: str, question: str -> answer: str'
#gist = 'document: str -> summary: str'

#print(help(dspy.Signature))

"""
dspy.Predict(
    dspy.Signature(
        '''answer the question'''

        question: str=dspy.InputField(desc="Question to be answered"),
        answer: Literal["yes", "no", "maybe"]=dspy.OutputField(desc="Answer to the question, either yes, no or maybe")
    )   
    )
"""

class CheckCitationFaithfulness(dspy.Signature):
    """Verify that the text is based on the provided context."""

    context: str = dspy.InputField(desc="facts here are assumed to be true")
    text: str = dspy.InputField()
    faithfulness: bool = dspy.OutputField()
    evidence: dict[str, list[str]] = dspy.OutputField(desc="Supporting evidence for claims")

context = "The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."

text = "Lee scored 3 goals for Colchester United."

faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)
'''
print(faithfulness(context=context, text=text))
print(type(faithfulness(context=context, text=text).faithfulness))
print(type(faithfulness(context=context, text=text).reasoning))
print(type(faithfulness(context=context, text=text).evidence))
'''

class Place:
    class AreaCovered(pydantic.BaseModel):
        area: float
        perimeter: float
    class QueryResult(pydantic.BaseModel):
        text: str
        score: float

signature = dspy.Signature("query: str -> area: Place.AreaCovered, result: Place.QueryResult")
print(signature)
print(type(signature))
module = dspy.Predict(signature)
print(module(query="chennai"))
print(type(module(query="chennai").result))
import dspy
import asyncio
from dotenv import dotenv_values, load_dotenv

load_dotenv()

config=dotenv_values("..\.env")
endpoint = config['AZURE_OPENAI_ENDPOINT']
openai_api_key = config['AZURE_OPENAI_API_KEY']
openai_api_version = config['AZURE_OPENAI_API_VERSION']
deployment_id=config['DEPLOYMENT_ID']

lm = dspy.LM('azure/gpt-4o-mini',
                    deployment_id=deployment_id,
                    api_key=openai_api_key,
                    api_base=endpoint,
                    model_type="chat",
                    api_version=openai_api_version)

dspy.settings.configure(lm=lm, trace=["Test"])


def search_wikipedia(query: str) -> list[str]:
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=2)
    return [x['text'] for x in results]

class Distance(dspy.Signature):
    """Return distance between two places"""
    place_1: str = dspy.InputField(desc="Starting point of journey")
    place_2: str = dspy.InputField(desc="Destination point of journey")
    distance: str = dspy.OutputField(desc="Answer in JSON {Start: place_1, Destination: place_2\nDistance in float}")


class ChitChat(dspy.Signature):
    """You are a personal companinon to spend time making chitchat and jovial coversations in thanglish"""
    convo: str = dspy.InputField(desc="Interaction from user who's sentiment must be analysed")
    reply: str = dspy.OutputField(desc="Thanglish reply in markup text with facial reactions like Wink, Smile, shrug, etc")

class Tutor(dspy.Signature):
    """You are a tutor to design roadmap for technology learning"""
    tech: str = dspy.InputField(desc="Technology needed to be learned")
    time: str = dspy.InputField(desc="Time user can maximum spend to learn the technology")
    level: str = dspy.InputField(desc="Technology level user needs to master")
    rm: str = dspy.OutputField(desc="Roadmap to learn the level of technology within the specified time, along with a flowchart")


dis = dspy.Predict(Distance)
chat = dspy.Predict(ChitChat)
tutor = dspy.ReAct(Tutor, tools=[search_wikipedia])

#content = tutor(tech="cloud computing", time="25 days", level="advanced")

#print(type(content.rm))
#print(tutor)
#print(dis(place_1="Pollachi",place_2="Coimbatore"))
#print(chat)


stream_predict = dspy.streamify(
    tutor,
    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="rm")],
)


async def read_output_stream():
    output_stream = stream_predict(tech="python", time="10 days", level="medium")

    async for chunk in output_stream:
        if isinstance(chunk, dspy.streaming.StreamResponse):
            #return_value = chunk
            print(chunk.chunk,end='')
        #return return_value


asyncio.run(read_output_stream())

import dspy
from dotenv import dotenv_values, load_dotenv

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

#print("output : ",lm("tell me about ai"))  #api output [messages, cache, max_tokens, temperature, top_p, top_k, stop, stream, stream_listeners, trace]
#print("Type of output : ", type(lm("tell me about ai")))  #list
#print("Length : ", len(lm("tell me about ai")))  #length of output list is 1
#print("First index : ", lm("tell me about ai")[0])

dspy.configure(lm=lm)
#use dspy.configure for global configuration
#use dspy.context for local configuration

output= lm(messages=[{"role":"assistant","content":"what is computer?"}])
#roles = ["system", "user", "assistant", "function", "developer", "tool"]
#messages-kwargs = [role, content, name, function_call, tool_calls, tool_call_id, tool_call_response]

print(lm.history)   
print(lm.history[-1].keys()) 
'''
    output 
    dict_keys([
        'prompt', 'messages', 'kwargs', 
        'response', 'outputs', 'usage', 
        'cost', 'timestamp', 'uuid', 'model', 
        'response_model', 'model_type'])
'''
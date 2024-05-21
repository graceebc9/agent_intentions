from pathlib import Path
import os
import json
import time
import anthropic
from anthropic import HUMAN_PROMPT, AI_PROMPT
from openai import OpenAI
import tiktoken
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from together import Together


def metadata2string(metadata: dict, enable_negation: bool = False) -> [str, str]:
    #
    # Convert metadata (dictionary structure) to a string in the form of a sentence.
    #
    # INPUTS:
    #   metadata (dict)         metadata to convert (in the form of the metadata in the Leap Of Thought data sets.
    #   enable_negation (bool)  If consider_negation is True and the "validity" of the statement is false, negate the string.
    #                           (Defaults to False, no negation.)
    #
    output = "A " + metadata["subject"] + metadata["predicate"] + metadata["object"] + "."
    output = output.replace("/r/IsA"," is a ")
    output = output.replace("hypernym"," is a ")
    #output = output.replace("/r/HasA"," has a ")
    output = output.replace("meronym"," has a ")
    output = output.replace("/r/PartOf"," is part of ")
    output = output.replace("/r/Antonym"," is not a ")
    output = output.replace("/r/Desires"," desires ")
    output = output.replace("/r/CapableOf"," is capable of ")

    # If necessary to negate the sentence, use these lines
    if enable_negation and (metadata["validity"] == "never true"):
        output = output.replace(" has ", " does not have ")
        output = output.replace(" desires ", " does not desire ")
        output = output.replace(" is ", " is not ").replace(" is not not ", " is ")

    if metadata["validity"] not in ["always true", "never true"]:
        raise Exception("Odd validity:", validity)
    return output, str(int(metadata["validity"]=="always true"))


def query_model(model: str, input: str, temperature: float=0.0) -> str:
    # 
    # Query the specified model with the specified input sting and return the one-token response.
    #

    if model == "davinci-002":
        input = statement2completion(input)
        client = OpenAI(organization = os.environ.get("OPENAI_ORG_KEY"))
        completion = client.completions.create(
            model=model,
            temperature = temperature,
            max_tokens=10,
            #logit_bias = davinci002_logitbias_10_json,
            logit_bias = davinci002_logitbias_truefalse_json,
            prompt = input
            )
        reply_content = completion.choices[0].text.strip()
        return allow_truefalse_yesno(reply_content)

    # OpenAI models
    elif model in openai_models:
        #print("OPENAI MODEL = ", model)
        emphasize10_json = {"16": 100, "15": 100}
        emphasize_truefalse_json = {"2575": 100, "4139": 100}
        client = OpenAI(organization = os.environ.get("OPENAI_ORG_KEY"))
        completion = client.chat.completions.create(
            model=model,
            temperature = temperature,
            max_tokens = 1,
            #logit_bias = gpt_logitbias_10_json,
            logprobs = False,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": statement2query(input)}
                ]
            )
        return completion.choices[0].message.content.strip()

    # Anthropic models
    elif model in anthropic_models:
        #print("ANTHROPIC MODEL = ", model)
        client = anthropic.Anthropic(
            # defaults to os.environ.get("ANTHROPIC_API_KEY") which is set as a user environment variable
            )
        message = client.messages.create(
            model=model,
            temperature = temperature,
            max_tokens= 2,
            messages=[{"role": "user", "content": statement2query(input)}]
            )
        time.sleep(1.3)
        return message.content[0].text

    # Mistral models
    elif model in mistral_models:
        client = MistralClient(api_key=os.environ.get("MISTRAL_API_KEY"))
        chat_response = client.chat(
            model=model,
            max_tokens = 2,
            temperature = temperature,
            messages = [ChatMessage(role="user", content = statement2completion(input)) ]
            )
        response = chat_response.choices[0].message.content.strip(" ")
        time.sleep(0.21)
        return allow_truefalse_yesno(response)

    # Together.ai Mistral models
    elif model in together_mistral_models:
        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        completion = client.chat.completions.create(
            model=model,
            temperature = temperature,
                max_tokens = 2,
            messages=[{"role": "user", "content": statement2completion(input)}],
        )
        response = completion.choices[0].message.content.strip(" ")
        time.sleep(1.05)
        return allow_truefalse_yesno(response)
    
    # If model is not in the known lists, raise an exception
    raise Exception("UNKNOWN MODEL:", model)


# Return 1 or 0 if the input contains yes, no, true, or false
def allow_truefalse_yesno(input):
    input_lower = input.lower()
    if ("true" in input_lower) or ("yes" in input_lower):
        return "1"
    elif ("false" in input_lower) or ("no" in input_lower):
        return "0"
    else:
        return input


# Convert a statement into a true/false verification in completion form.
def statement2completion(input:str) -> str:
    return "Complete only with one word, either true or false. " + input + " The preceding statement is... "


# Convert a statement into a true/false verification query.
def statement2query(input:str) -> str:
    return "Is the following true? " + input + " Answer only '1' for yes or '0' for no."


###########################################################################
# Query all test phrases in the data set
def main(test_filename: str, model: str, num_tests: int, temperature: float=0.0):
    #
    # Query all the test statements: base_property, entailing_statement (if specified), entailed statement.
    # Write the results out to a text file.
    #
    # INPUT:
    #   test_filename (string)   name of the data file, assumed to be a .json file
    #   model (string)      which model to use
    #   num_tests (int)     number of tests to run
    #   temperature (float) temperature fed to the model (0 = consistent, 1 = more random)
    #                           defaults to 0 (max consistency)

    tests, output = [], []
    with open(test_filename, 'r') as jsonfile:
        for line in jsonfile:
            tests.append(json.loads(line))
    jsonfile.close()
    tests = tests[: num_tests] # Limit number of tests we perform

    # Test all base properties
    for i, test in enumerate(tests):
        base_property = test["metadata"]["property"]
        base_property_string, base_property_true_answer = metadata2string(base_property)
        #print("BASE_PROPERTY:",base_property_string)
        base_property_model_answer = query_model(model, base_property_string, temperature)
        if base_property_model_answer not in ["1", "0"]:
            print("BASE_PROPERTY:",base_property_string)
            print(base_property_model_answer,"\n")
        output.append([[None,None], [None,None], [None,None]])
        output[i][0] = [base_property_true_answer[0], base_property_model_answer[0]]
        
    # Test all entailing properties
    for i, test in enumerate(tests):
        if "implicit_rule" in test["metadata"].keys():
            entailing_statement = test["metadata"]["implicit_rule"]
            entailing_statement_string, entailing_statement_true_answer = metadata2string(entailing_statement)
            entailing_statement_model_answer = query_model(model, entailing_statement_string, temperature)
            if entailing_statement_model_answer not in ["1", "0"]:
                print("ENTAILING STATEMENT:", entailing_statement_string)
                print(entailing_statement_model_answer,"\n")
            output[i][1] = [entailing_statement_true_answer[0], entailing_statement_model_answer[0]]
        else:
            base_property = test["metadata"]["property"]
            base_property_string, base_property_true_answer = metadata2string(base_property)
            entailed_property = test["metadata"]["statement"]
            entailed_property_string, entailed_property_true_answer = metadata2string(entailed_property)
            #print("Entailing statement not explicitly given.", base_property_string, entailed_property_string)
            output[i][1] = ["x", "x"]
        
    # Test all entailed properties
    for i, test in enumerate(tests):
        entailed_property = test["metadata"]["statement"]
        entailed_property_string, entailed_property_true_answer = metadata2string(entailed_property)
        entailed_property_model_answer = query_model(model, entailed_property_string, temperature)
        if entailed_property_model_answer not in ["1", "0"]:
            print("ENTAILED_PROPERTY:",entailed_property_string)
            print(entailed_property_model_answer,"\n")
        output[i][2] = [entailed_property_true_answer[0], entailed_property_model_answer[0]]
        
    # Write the answers to a text file.
    output_filename = test_filename.strip(".jsonl") + "_" + model.split("/")[-1] + ".txt"
    output_file = open(output_filename, "w")
    for tuple in output:
        list = [".".join(pair) for pair in tuple]
        output_string = " ".join(list)
        output_file.write(output_string + "\n")
    output_file.close()
    return



##########################################################################
anthropic_models = ["claude-instant-1.2", "claude-2.0", "claude-2.1", "claude-3-haiku-20240229", "claude-3-sonnet-20240307", "claude-1-opus-20240229"]
#openai_models = ["davinci-002", "gpt-3.5-turbo", "gpt-4"]
openai_models = ["gpt-3.5-turbo", "gpt-4"]
mistral_models = ["open-mistral-7b", "open-mixtral-8x7b", "open-mixtral-8x22b"]
together_mistral_models = ["mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mixtral-8x7B-Instruct-v0.1", "mistralai/Mixtral-8x22B-Instruct-v0.1"]

test_filename = "data_hypernyms_hypernyms_explicit_only_short_neg_hypernym_rule_test.jsonl"

if __name__ == "__main__":
    models_to_test = mistral_models 
    temperature = 0
    num_tests = 9999 

    # Set up all the logit_bias json structures
    encoding = tiktoken.encoding_for_model("gpt-4")
    gpt_logitbias_truefalse_json = {encoding.encode("True")[0]: 100, encoding.encode("False")[0]: 100}
    gpt_logitbias_10_json = {encoding.encode("1")[0]: 100, encoding.encode("0")[0]: 100}
    encoding = tiktoken.encoding_for_model("davinci-002")
    davinci002_logitbias_truefalse_json = {encoding.encode("True")[0]: 100, encoding.encode("False")[0]: 100}
    davinci002_logitbias_10_json = {encoding.encode("1")[0]: 100, encoding.encode("0")[0]: 100}

    # Query all the test phrases and save the results
    for model in models_to_test:
        print(f"\n{model}")
        main(test_filename, model, num_tests, temperature)

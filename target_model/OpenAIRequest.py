import openai
import json
from Secret import SECRET_KEY

def read_article_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_results_to_file(results, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

def ask_openai(text, category):
    prompt = f"Given the following text, list all entities related to the category '{category}' in a JSON array format. If no entities related to the category are found, return an empty array ([]).\n\nText: {text}\n\nCategory: {category}\n\nEntities:"
    try:
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt,
            temperature=0,
            max_tokens=250,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=None
        )
       
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        return "[]"

def ask_openai_allEntities(text):
    prompt = f"""An entity is a category that can be any of the following: organization, person, product, location, event, currency, rate, date, law, stock ticker, indicator, metric, quantity, commodity, or monetary value. 
    \n Given the provided text, extract every entity mentioned along with its corresponding values and return them in a JSON array format. 
    \n Each element in the JSON array should represent a unique entity category, containing two fields: 'entity', which is a string representing the entity category, and 'value', which is an array storing strings of all values belonging to the entity as referenced in the text.
    \n The final output should be a JSON array containing JSON objects, 
    \n with each object representing an entity and its associated values. If no entities related to a category are found in the text, return an empty array ([]). 
    \n\nText: {text}"""
    try:
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt,
            temperature=1.0,
            max_tokens=1500,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=None
        )
        json_response = response.choices[0].text.strip()
        index_of_start = json_response.find("[")
        json_response.replace('""""', '"')
        if '"entity":' in  json_response[0:len('"entity":')]:
            json_response = "[{" + json_response 
        if index_of_start != 0:
            json_response = json_response[index_of_start:]
        if '"entity"' not in json_response:
            json_response.replace('entity:', '"entity":')
            json_response.replace('value:', '"value":')
        if "'values'" in json_response:
            json_response.replace("'values'", "'value'")
        if '"values"' in json_response:
            json_response.replace("'values'", "'value'")
        if json_response[-1] != "]":
            if json_response[-1] == '}':
                json_response = json_response + "]"
            if json_response[-1] == ',':
                json_response[-1] = "}"
                json_response = json_response + "]"
        return json_response
    except Exception as e:
        print(f"An error occurred: {e}")
        return "[]"

def get_GPTInstruction(text, instruction_list):
    openai.api_key = SECRET_KEY

    #get_GPTInstruction(text, output_file_path)
    entity_categories = ["organization", "person", "product", "location", "event", "currency", "rate", "date", "law", "stock ticker", "indicator", "quantity", "commodity", "monetary value", "metric"]
    answer = ask_openai_allEntities(text)
    # Ensure GPT's answer is formatted correctly as a JSON array
    try:
        # Attempt to parse the answer to validate it's a proper JSON format
        json.loads(answer)
        answer_data = json.loads(answer)
    except ValueError as e:
        # If parsing fails, set answer to an empty array
        answer_data = []
    answer_dict = {}
    print(answer_data)
    try:
        if answer_data != "[]":
            for item in answer_data:
                if 'entity' in item.keys() and 'value' in item.keys():
                    key = item['entity']
                    #check if entity is in answer_dict
                    if key in answer_dict.keys():
                        #get the entity name for item
                        #append the value list to the entity in the answer dictionary 
                        value = item['value']
                        list_set = set(value)
                        # convert the set to the list
                        unique_value = (list(list_set))
                        for val in unique_value:
                            if val not in answer_dict[key]:
                                answer_dict[key].append(val)
                    else:
                        value = item['value']
                        list_set = set(value)
                        # convert the set to the list
                        unique_value = (list(list_set))
                        answer_dict[key] = unique_value
    except Exception as e:
        print("found error skipping")
    conversation_flow = create_convo_flow(entity_categories, answer_dict, text, instruction_list) 
    return conversation_flow

def create_convo_flow(entity_categories, answer_dict, text, instruction_list):
    print(answer_dict.items())
    for key,value in answer_dict.items():
        conversation_flow = [{"from": "human", "value": text}, {"from": "gpt", "value": "I've read this text."}]
        conversation_flow.append({"from": "human", "value": f"What describes {key} in the text?"})
        conversation_flow.append({"from": "gpt", "value": value})
        id = "NER_" + str(len(instruction_list))
        instruction_list.append({"id": id, "conversations": conversation_flow})
        if key in entity_categories: 
            entity_categories.remove(key)

    #initialize the other categories to []
    for category in entity_categories:
        conversation_flow = [{"from": "human", "value": text}, {"from": "gpt", "value": "I've read this text."}]
        conversation_flow.append({"from": "human", "value": f"What describes {category} in the text?"})
        conversation_flow.append({"from": "gpt", "value": []})
        id = "NER_" + str(len(instruction_list))
        instruction_list.append({"id": id, "conversations": conversation_flow})
    return instruction_list


if __name__ == "__main__":
    openai.api_key = "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"
    input_file_path = 'target_model/input_article.txt'
    output_file_path = 'target_model/output_entities.json'

    text = read_article_from_file(input_file_path)
    conversation_flow = get_GPTInstruction(text)


    entity_categories = ["law", "organization", "person", "object", "function", "compound", "location", "chemical", "profession"]

 
    write_results_to_file(conversation_flow, output_file_path)

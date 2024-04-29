import OpenAIRequest
import pandas as pd
import raw_data
import spacy
import os
import json
from Secret import SECRET_KEY

def generate_instructions(input_file, instruction_file= None, instruction_folder = None):
    instruction_df = pd.DataFrame()
    instruction_df["id"] = []
    instruction_df["conversations"] = []
    instruction_list = []
    if ".csv" in input_file:
        df = pd.read_csv(input_file)
        for text in df["Text"]:
            conversation = OpenAIRequest.get_GPTInstruction(text)
            id_key = "ner_gpt_"+str(count)
            new_row = {'id': id_key, "conversations": conversation}
            instruction_df.loc[len(instruction_df)] = new_row
    if ".json" in input_file:
         series = pd.read_json(input_file,  typ='series')
         df =  pd.DataFrame(series).transpose()
         for col in df.columns:
             if "item" in col:
                doc = nlp(df[col].iloc[0])
                sentences = list(doc.sents)
                count = 1
                while len(sentences) > 10:
                    #request GPT first 10 sentences 
                    sent_text = [sent.text for sent in sentences[0:10]]
                    text = " ".join(sent_text)
                    instruction_list = OpenAIRequest.get_GPTInstruction(text, instruction_list)
                    sentences = sentences[11:]

                if len(sentences) > 0:
                    sent_text = [sent.text for sent in sentences]
                    text = " ".join(sent_text)
                    instruction_list = OpenAIRequest.get_GPTInstruction(text, instruction_list)

    if instruction_file:
        with open("target_model/instruction_data/" + instruction_folder + "/" + instruction_file+ "_instructions.json", 'w') as f:
            json.dump(instruction_list, f)
        
    else:
        instruction_df["conversations"].to_json("target_model/instruction_data/sorted/"+ "sec_instructions.json")
        
if __name__ == "__main__":
    nlp = spacy.load('en_core_web_sm')
    folder_path = "target_model/raw_data/SEC_filings_2021/"
    instruction_file = "test"

    for root, directories, files in os.walk(folder_path):
        file_names = []
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                file_names.append(file_path)
    count = 1
    file_names.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    for file_path in file_names:
        if count >=108 and count <= 150:
            print(count)
            print(file_path)
            generate_instructions(file_path, "sec_" + str(count), "training_1")
        if count > 150:
            generate_instructions(file_path, "sec_" + str(count), "testing_1")
        count+=1

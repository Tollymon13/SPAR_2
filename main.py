from pathlib import Path
import re
import random
import json
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).parent
data_location_e1_biathlon = (BASE/"msa-cogsci-2025-data-main"/"example-scenarios"/"e1-e2/").resolve()
data_location_frames = (BASE/"msa-cogsci-2025-data-main"/"msa-frame-prompts/").resolve()
#  Path(data_location_frames/filepath)
# Path(data_location_e1_biathlon/filepath)

class AssembleText:
    # combine multiple texts together before replacing
    # when we assemble, we can determine which texts to use i.e. exclude the current one being prompted
    def __init__(self, parts: list):
        self.parts = parts 
    
    def concat_text(self, start_text, end_text, separator="\n"):
        temp_text = []
        for p in self.parts:
            self.match = re.search(rf"{start_text}(.*?){end_text}", p, re.DOTALL)
            self.sub_text = self.match.group(1).strip()
            temp_text.append(self.sub_text)
        self.final_concat = separator.join(temp_text) 
        return self.final_concat
    
    def __str__(self):
        return self.final_concat

class StringManipulator:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.frame = self.filepath.read_text(encoding="utf-8")

    def replace_text(self, search_text, replacing_text):
        self.frame = self.frame.replace(search_text, replacing_text)
        return self
    
    def __str__(self):
        return self.frame

class LLM():
    def __init__(self, model="meta-llama/llama-3.3-70b-instruct", max_tokens=1000):
        self.client = OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = model
        self.max_tokens = max_tokens

    def prompt(self, user_message, system_message=None):
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens
        )

        return response.choices[0].message.content


def shuffler(input_list, current_example):
    remaining_examples = [k for k in input_list if k != current_example]
    shuffled = random.sample(remaining_examples, len(remaining_examples)) 
    return shuffled

def read_scenarios(filepath):
    list_scenarios_paths = [f for f in filepath.iterdir() if f.is_file()]
    list_read_scenarios = [Path(f).read_text(encoding="utf-8") for f in list_scenarios_paths]
    return list_read_scenarios

def read_examples(filepath):
    biathlon_example = Path(filepath/"biathlon.txt")
    canoe_example = Path(filepath/"canoe-racing.txt")
    diving_example = Path(filepath/"diving.txt")
    tug_of_war_example = Path(filepath/"tug-of-war.txt")
    exam_example = Path(filepath/"exam.txt")
    list_examples_paths = [biathlon_example, canoe_example, diving_example, tug_of_war_example, exam_example]
    return [f.read_text(encoding="utf-8") for f in list_examples_paths]

if __name__ == '__main__':
    BASE = Path(__file__).parent
    data_location_e1_e2 = (BASE/"msa-cogsci-2025-data-main"/"example-scenarios"/"e1-e2/").resolve()
    data_location_frames = (BASE/"msa-cogsci-2025-data-main"/"msa-frame-prompts/").resolve()
    data_location_scenarios_e1 = (BASE/"msa-cogsci-2025-data-main"/"model-olympics-human-experiment"/"e1"/"scenarios/").resolve()
    list_e1_scenarios = read_scenarios(data_location_scenarios_e1)
    list_examples_e1_e2 = read_examples(data_location_e1_e2)
    parsing_frame = Path(data_location_frames/"generate-parsing.txt")
    system_frame = Path(data_location_frames/"generate-system-prompt.txt")
    shuffled_examples = shuffler(list_examples_e1_e2, list_examples_e1_e2[0])

    concat_variables = AssembleText(shuffled_examples)
    concat_text = concat_variables.concat_text("<START_SCENARIO>", "<END_LANGUAGE_TO_WEBPPL_CODE>")
    user_prompt = StringManipulator(parsing_frame)
    user_prompt.replace_text("<SHUFFLED EXAMPLES OF SCENARIOS AND START_LANGUAGE_TO_WEBPPL_CODE DELIMITED BLOCK INJECTED HERE>", concat_text)
    user_prompt = user_prompt.replace_text("<SCENARIO_INJECTED_HERE>", list_e1_scenarios[0])
    system_prompt = StringManipulator(system_frame)
    llm = LLM()
    output = llm.prompt(str(user_prompt), str(system_prompt))
    print(output)

    # create the API calls nows


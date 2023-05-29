import json
import os
import pandas as pd

def load_json_data(path):
    print(f"Loading jsond data from: {path}")
    json_data = []
    lines = []
    with open(path) as f:
        s = f.readlines()
    
    if len(s) == 1:
        with open(path + ".backup", "w+") as f_backup:
            f_backup.write(s[0])

        print(f"OOPS! {path} is not a valid jsonl file. Found 1 line Will add breaks")
        s_new = s[0].replace("}{", "}\n{")

        with open(path, "w+") as f2:
            f2.write(s_new)

    with open(path) as f:
        lines = f.readlines()
        print(f"Found {len(lines)} lines")
        for ln in lines:
            json_data.append(json.loads(ln))
    return json_data



if __name__ == "__main__":
    split = "val"
    path_stub = "/home/ubuntu/RL4LMs/rl4lm_exps/image_prompts_gpt2_ppo.yml/image_generations"
    
    ts_list = os.listdir(os.path.join(path_stub, split)) 
    ts_list_int = [int(s) for s in ts_list]
    first_ts = str(min(ts_list_int))
    last_ts = str(max(ts_list_int))

    first_data_path = os.path.join(path_stub, split, first_ts, "generation_data.jsonl")
    last_data_path = os.path.join(path_stub, split, last_ts, "generation_data.jsonl")
    
    first_data = load_json_data(first_data_path)
    last_data = load_json_data(last_data_path)

        
    similarities_dict = {}
    for it in first_data:
        similarities_dict[it["image_id"]] = {first_ts: it["image_similarity_score"]}

    for it in last_data:
        similarities_dict[it["image_id"]][last_ts] = it["image_similarity_score"]

    df = pd.DataFrame.from_dict(similarities_dict, orient="index")
    df["diff"] = df[last_ts] - df[first_ts]
    df = df.sort_values(by=["diff"])

    import pdb;pdb.set_trace()




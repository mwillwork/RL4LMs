from transformers import ConvNextForImageClassification
import json

def filter_imagenet():
    model_name_or_path = "facebook/convnext-base-224"
    model = ConvNextForImageClassification.from_pretrained(model_name_or_path)
    output_filename = "/home/ubuntu/RL4LMs/imagenet_filtered.jsonl"
    with open(output_filename, "a") as o_file:
        cnt = 0
        for k, v in model.config.id2label.items():
            topic = v.split(",")[0]
            obj = {
                "is_esoteric": 1,
                "topic": topic,
                "imagenet_idx": k,
                "imagenet_full_label": v
            }
            cnt += 1
            print("NOT WRITING BECAUSE WOULD APPEND")
            # o_file.write(json.dumps(obj) + "\n")
    print(f"output: {output_filename}, wrote {cnt} classes")


def test_load_imagenet():
    filename = "/home/ubuntu/RL4LMs/imagenet_filtered.jsonl"
    with open(filename, "r") as f:
        lines = f.readlines()
        cnt = 0
        for ln in lines:
            json_data = json.loads(ln)
            if json_data["is_esoteric"] == 0:
                print(json_data)
                cnt += 1
        print(f"Found {cnt} non-esoteric classes.")

if __name__ == "__main__":
    #filter_imagenet()
    test_load_imagenet()

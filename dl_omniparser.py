from huggingface_hub import hf_hub_download

revision = "3ad04208583386e8ad1ac2505d245200eee29efd"

for filename in ["model.pt", "model.yaml", "best-009-0.7396-ema.pt", "LICENSE"]:
  path = hf_hub_download(
    repo_id="microsoft/OmniParser-v2.0",
    filename="icon_detect_v3/" + filename,
    revision=revision,
    local_dir="weights/omniparser",
  )
  print("Downloaded:", path)

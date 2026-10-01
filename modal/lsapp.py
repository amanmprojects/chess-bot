import modal
app = modal.App("lsvol")
vol = modal.Volume.from_name("chess-bot-data")

@app.function(image=modal.Image.debian_slim(), volumes={"/vol": vol})
def ls():
    import os
    vol.reload()
    return os.listdir("/vol")

@app.local_entrypoint()
def main():
    print("LISTING:", ls.remote())

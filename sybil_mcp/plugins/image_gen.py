import openai
import os

class ImageGenPlugin:
    name = "image_gen"

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set.")
        openai.api_key = self.api_key

    def generate_image(self, prompt, user_id=None):
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        return response['data'][0]['url']

def register_plugin():
    return ImageGenPlugin() 
class MCPPluginBase:
    name: str
    def generate_image(self, prompt: str, user_id: str = None):
        raise NotImplementedError 
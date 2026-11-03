from openai import OpenAI


class LLMExtension:
    client: OpenAI | None = None

    def __init__(self, app=None):
        if app is not None:
            self.init_app(app)

    def init_model(self, app):
        api_key = app.config.get("OPENAI_API_KEY", None)
        return OpenAI(api_key=api_key)

    def init_app(self, app):
        self.client = self.init_model(app)
        app.extensions = getattr(app, "extensions", {})
        app.extensions["client"] = self.client

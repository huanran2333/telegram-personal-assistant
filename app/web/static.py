"""Static workbench delivery: HTML must not pin an obsolete auth bootstrap."""

from fastapi.staticfiles import StaticFiles


class WorkbenchStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if response.media_type == "text/html":
            response.headers["Cache-Control"] = "no-store"
        return response

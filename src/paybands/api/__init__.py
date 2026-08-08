"""The FastAPI service — Layer 1 and Layer 2, behind three endpoints.

Nothing is imported here on purpose. `app.py` pulls in FastAPI and `service.py`
pulls in LightGBM, and neither should be a cost paid by anyone who only wanted
`paybands.api.models` to read the wire contract. Import the module you need:

    from paybands.api.app import app                  # the ASGI application
    from paybands.api.service import train_bundle     # train a local artifact
"""

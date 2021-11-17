from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

import dataloader.binance_data as dlb
import hydra.Environments

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def index():
    return """
        <html>
            <head>
                <title>Some HTML in here</title>
            </head>
            <body>
                <h1>Look ma! HTML!</h1>
            </body>
        </html>
        """


@app.get("/fft")
def get_fft():
    return None  # load existing fft if exists else throw errors?


@app.post("/fft")
def make_fft():
    """
    # create FFT & save it
    Save parameters + time range + window
    Future access will retrieve by time range and window
    May save as pickle, with retrieval indexed using file names
    Or maybe just set up a basic DB?
    """
    return get_fft()
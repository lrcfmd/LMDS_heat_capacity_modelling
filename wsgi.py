import os
from site import getsitepackages
from app import app

wd = os.getcwd()
sitepack = getsitepackages()

if __name__ == "__main__":
    app.run()

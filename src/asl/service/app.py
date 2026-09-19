import time
import uuid

from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException

from pydantic import BaseModel, Field

from asl import db
from asl.config import settings


class Features(BaseModel):
    model_config = 
from pydantic import BaseModel, conlist
from typing import Union


class Data(BaseModel):
    data: conlist(float, min_length=2, max_length=2)
    run_id: Union[str, None] = None

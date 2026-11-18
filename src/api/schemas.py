from pydantic import BaseModel, Field
from typing import Optional


class Transaction(BaseModel):
    transaction_id: str = Field(alias="TransactionID")
    TransactionDT: Optional[float] = None
    TransactionAmt: Optional[float] = None
    card1: Optional[str] = None
    DeviceID: Optional[str] = None
    P_emaildomain: Optional[str] = None

    model_config = {"populate_by_name": True}

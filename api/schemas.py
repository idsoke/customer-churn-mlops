from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerData(BaseModel):
    """Data pelanggan mentah, format sama seperti dataset asli (sebelum encoding)."""

    Tenure: float = Field(..., ge=0, description="Lama berlangganan (bulan)")
    PreferredLoginDevice: Literal["Mobile Phone", "Phone", "Computer"]
    CityTier: int = Field(..., ge=1, le=3)
    WarehouseToHome: float = Field(..., ge=0, description="Jarak gudang ke rumah (km)")
    PreferredPaymentMode: Literal["Credit Card", "Debit Card", "E wallet", "Cash on Delivery", "UPI"]
    Gender: Literal["Male", "Female"]
    HourSpendOnApp: float = Field(..., ge=0, description="Rata-rata jam di aplikasi per hari")
    NumberOfDeviceRegistered: int = Field(..., ge=0)
    PreferedOrderCat: Literal["Mobile Phone", "Fashion", "Grocery", "Laptop & Accessory", "Others"]
    SatisfactionScore: int = Field(..., ge=1, le=5)
    MaritalStatus: Literal["Single", "Married", "Divorced"]
    NumberOfAddress: int = Field(..., ge=0)
    Complain: int = Field(..., ge=0, le=1, description="1 jika pernah komplain, 0 jika tidak")
    OrderAmountHikeFromlastYear: float = Field(..., ge=0, description="Kenaikan nilai pesanan dari tahun lalu (%)")
    CouponUsed: int = Field(..., ge=0)
    OrderCount: int = Field(..., ge=0)
    DaySinceLastOrder: float = Field(..., ge=0)
    CashbackAmount: float = Field(..., ge=0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Tenure": 12,
                "PreferredLoginDevice": "Mobile Phone",
                "CityTier": 1,
                "WarehouseToHome": 15,
                "PreferredPaymentMode": "Credit Card",
                "Gender": "Male",
                "HourSpendOnApp": 3.5,
                "NumberOfDeviceRegistered": 4,
                "PreferedOrderCat": "Laptop & Accessory",
                "SatisfactionScore": 3,
                "MaritalStatus": "Single",
                "NumberOfAddress": 2,
                "Complain": 0,
                "OrderAmountHikeFromlastYear": 15,
                "CouponUsed": 1,
                "OrderCount": 2,
                "DaySinceLastOrder": 5,
                "CashbackAmount": 150.5,
            }
        }
    )


class PredictionResponse(BaseModel):
    churn: bool
    churn_probability: float
    risk_level: Literal["Tinggi", "Aman"]


class ModelInfo(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

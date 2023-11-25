from pydantic import BaseModel, Field
from pymongo import MongoClient
from bson import ObjectId
from typing import Optional, List

client = MongoClient()
db = client.fastshopping


class PyObjectId(ObjectId):

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError('Invalid ObjectId')
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type='string')


class Product(BaseModel):
    id: Optional[PyObjectId] = Field(alias='_id')
    name: str
    ratings: str
    image_path: str
    carousel: List[str]
    abstract_review: str

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str
        }
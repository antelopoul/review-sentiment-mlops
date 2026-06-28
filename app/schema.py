from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


# Shared types
JobStatus = Literal["queued", "running", "success", "failed", "timeout"]
EventType = Literal["job_start", "status_update", "job_result"]


# Shared sub-models
class JobInput(BaseModel):
    text: str


class JobMetadata(BaseModel):
    helpful_vote: Optional[str] = None
    verified_purchase: Optional[str] = None


# HTTP API — request / response
class SentimentRequestV1(BaseModel):
    schema_version: Literal["sentiment_request_v1"] = "sentiment_request_v1"
    text: str


class SentimentResponseV1(BaseModel):
    schema_version: Literal["sentiment_response_v1"] = "sentiment_response_v1"
    prediction: Literal["negative", "neutral", "positive"]
    label: int
    confidence: float
    logits: list[float]
    model_version: str
    request_id: str


class SentimentBatchRequestV1(BaseModel):
    schema_version: Literal["sentiment_batch_request_v1"] = "sentiment_batch_request_v1"
    texts: list[str]


class SentimentBatchResponseV1(BaseModel):
    schema_version: Literal["sentiment_batch_response_v1"] = "sentiment_batch_response_v1"
    results: list[SentimentResponseV1]
    model_version: str
    request_id: str


# Job Start
class SentimentJobStartV1(BaseModel):
    schema_version: Literal["sentiment_job_start_v1"] = "sentiment_job_start_v1"
    job_id: str
    event: Literal["job_start"]
    timestamp: datetime
    input: JobInput
    metadata: JobMetadata
    status: JobStatus = "queued"


# Job Status Update
class SentimentJobStatusV1(BaseModel):
    schema_version: Literal["sentiment_job_status_v1"] = "sentiment_job_status_v1"
    job_id: str
    event: Literal["status_update"]
    timestamp: datetime
    status: JobStatus
    progress: float = Field(ge=0.0, le=1.0)
    message: Optional[str] = None


# Job Result + metrics
class SentimentResult(BaseModel):
    label: Literal["positive", "neutral", "negative"]
    confidence: float


class SentimentMetrics(BaseModel):
    latency_ms: float
    model_latency_ms: float
    tokenization_ms: float


class SentimentModelInfo(BaseModel):
    name: str
    version: str


class SentimentError(BaseModel):
    type: str
    message: str


class SentimentJobResultV1(BaseModel):
    schema_version: Literal["sentiment_job_result_v1"] = "sentiment_job_result_v1"
    job_id: str
    event: Literal["job_result"]
    timestamp: datetime
    status: JobStatus

    result: Optional[SentimentResult] = None
    metrics: Optional[SentimentMetrics] = None
    model: Optional[SentimentModelInfo] = None
    error: Optional[SentimentError] = None

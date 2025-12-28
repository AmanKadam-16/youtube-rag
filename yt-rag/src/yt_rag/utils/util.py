from youtube_transcript_api import YouTubeTranscriptApi
from src.yt_rag.schema.ingest import TranscriptExtract
from src.yt_rag.core.config import settings
from src.yt_rag.core.llm.config import embedding_client
from pathlib import Path

"""
Response Format - 
{
    "text": "In doing this, the output tends to look a lot more natural if",
    "start": 73.08,
    "duration": 3.134
}

Schema
class TranscriptExtract(BaseModel):
    text: str
    start: float
    duration: float
"""


# https://www.youtube.com/watch?v=LPZh9BOjkQs
def transcript_extractor(video_url: str) -> list[TranscriptExtract]:
    video_id = video_url.split("?v=")[1]
    print(f"Video Id extracted : {video_id}")
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id)
    detailed_info = fetched_transcript.to_raw_data()
    result = []
    for transcript in detailed_info:
        new_entry = TranscriptExtract(
            text=transcript["text"],
            start=transcript["start"],
            duration=transcript["duration"],
        )
        result.append(new_entry)
    return result


# CHUNKING STRATEGY | FIXED LENGTH
def chunk_transcript(
    transcript_info: list[TranscriptExtract], chunk_size: int = 500, over_lap: int = 5
) -> list[str]:
    print("Step 02")
    print("Transcript Chunking in Progress...")
    seed_chunk = transcript_info[0].text
    result = [seed_chunk]
    for transcript in transcript_info:
        last_chunk = result[-1]
        current_text = transcript.text
        if not transcript:
            continue
        if len(last_chunk + current_text) <= chunk_size:
            result[-1] += f" {current_text}"
        else:
            result.append(last_chunk.split(" ")[-over_lap] + " " + current_text)
    return result


def create_embeddings(chunks_list: list[str]) -> list[dict]:
    print("Embedding Creation in Progress...")
    client = embedding_client
    response = client.embeddings.create(
        input=chunks_list,
        model=settings.EMBEDDING_MODEL,
        encoding_format="float",
        extra_body={"input_type": "passage", "truncate": "NONE"},
    )
    return response.data


def create_query_embedding(chunks_list: list[str]) -> list[float]:
    print("User Query Embedding Creation in Progress...")
    client = embedding_client
    response = client.embeddings.create(
        input=chunks_list,
        model=settings.EMBEDDING_MODEL,
        encoding_format="float",
        extra_body={"input_type": "passage", "truncate": "NONE"},
    )
    return response.data[0].embedding


def export_mermaid_graph(app, output_path="graph_01_version.mmd"):
    graph = app.get_graph()
    mermaid_code = graph.draw_mermaid()

    output_path = Path(output_path)
    output_path.write_text(mermaid_code, encoding="utf-8")

    return output_path


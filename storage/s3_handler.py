import json
import os
import base64
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from cryptography.fernet import Fernet


# ── Encryption helpers ────────────────────────────────────────────────────────

def _get_fernet() -> Fernet:
    """
    Load the encryption key from the environment and return a Fernet instance.

    The key must be a URL-safe base64-encoded 32-byte value.
    Generate one with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    """
    raw_key = os.getenv("ENCRYPTION_KEY")
    if not raw_key:
        raise EnvironmentError(
            "ENCRYPTION_KEY is not set. "
            "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
    return Fernet(raw_key.encode())


def encrypt(data: bytes) -> bytes:
    """Encrypt raw bytes using Fernet (AES-128-CBC + HMAC-SHA256)."""
    return _get_fernet().encrypt(data)


def decrypt(token: bytes) -> bytes:
    """Decrypt a Fernet token back to raw bytes."""
    return _get_fernet().decrypt(token)


# ── S3 client ─────────────────────────────────────────────────────────────────

def _get_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def upload_document(content: str, doc_id: str, metadata: dict) -> bool:
    """
    Encrypt and upload a markdown document and its metadata to S3.

    Encryption is applied in two layers:
      1. Client-side: Fernet (AES-128-CBC + HMAC) before data leaves the machine.
      2. Server-side: AWS SSE-S3 (AES-256) applied by S3 at rest.

    Args:
        content:  Markdown content to store.
        doc_id:   Unique identifier / filename (without extension).
        metadata: Dict of metadata to store alongside the document.

    Returns:
        True if upload succeeded, False otherwise.
    """
    bucket = os.getenv("AWS_BUCKET_NAME")
    if not bucket:
        raise EnvironmentError("AWS_BUCKET_NAME is not set in environment.")

    client = _get_client()

    # Client-side encryption
    encrypted_content = encrypt(content.encode("utf-8"))
    encrypted_metadata = encrypt(json.dumps(metadata).encode("utf-8"))

    try:
        # ServerSideEncryption adds AWS SSE-S3 (AES-256) as a second layer
        client.put_object(
            Bucket=bucket,
            Key=f"docs/{doc_id}.enc",
            Body=encrypted_content,
            ContentType="application/octet-stream",
            ServerSideEncryption="AES256",
        )
        client.put_object(
            Bucket=bucket,
            Key=f"meta/{doc_id}.enc",
            Body=encrypted_metadata,
            ContentType="application/octet-stream",
            ServerSideEncryption="AES256",
        )
        return True
    except (BotoCoreError, ClientError) as e:
        print(f"[S3] Upload failed: {e}")
        return False


def download_document(doc_id: str) -> str | None:
    """
    Download and decrypt a markdown document from S3.

    Args:
        doc_id: Unique identifier / filename (without extension).

    Returns:
        Decrypted markdown content string, or None if not found.
    """
    bucket = os.getenv("AWS_BUCKET_NAME")
    client = _get_client()

    try:
        response = client.get_object(Bucket=bucket, Key=f"docs/{doc_id}.enc")
        encrypted = response["Body"].read()
        return decrypt(encrypted).decode("utf-8")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return None
        raise


def list_documents() -> list[str]:
    """
    List all stored document IDs in the S3 bucket.

    Returns:
        List of doc_id strings.
    """
    bucket = os.getenv("AWS_BUCKET_NAME")
    client = _get_client()

    try:
        response = client.list_objects_v2(Bucket=bucket, Prefix="docs/")
        keys = [obj["Key"] for obj in response.get("Contents", [])]
        return [k.replace("docs/", "").replace(".enc", "") for k in keys]
    except ClientError as e:
        print(f"[S3] List failed: {e}")
        return []


def upload_image(image_b64: str, doc_id: str, index: int) -> bool:
    """
    Encrypt and upload a single base64-encoded image to S3.

    Args:
        image_b64: Base64-encoded image string.
        doc_id:    Parent document ID.
        index:     Image index within the document.

    Returns:
        True if upload succeeded, False otherwise.
    """
    bucket = os.getenv("AWS_BUCKET_NAME")
    if not bucket:
        raise EnvironmentError("AWS_BUCKET_NAME is not set in environment.")

    client = _get_client()
    encrypted = encrypt(image_b64.encode("utf-8"))

    try:
        client.put_object(
            Bucket=bucket,
            Key=f"images/{doc_id}/{index}.enc",
            Body=encrypted,
            ContentType="application/octet-stream",
            ServerSideEncryption="AES256",
        )
        return True
    except (BotoCoreError, ClientError) as e:
        print(f"[S3] Image upload failed: {e}")
        return False


def download_image(doc_id: str, index: int) -> str | None:
    """
    Download and decrypt a base64-encoded image from S3.

    Args:
        doc_id: Parent document ID.
        index:  Image index within the document.

    Returns:
        Decrypted base64 image string, or None if not found.
    """
    bucket = os.getenv("AWS_BUCKET_NAME")
    client = _get_client()

    try:
        response = client.get_object(Bucket=bucket, Key=f"images/{doc_id}/{index}.enc")
        encrypted = response["Body"].read()
        return decrypt(encrypted).decode("utf-8")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return None
        raise
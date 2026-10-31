import mimetypes
import os
import posixpath
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import NamedTuple

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config


class uniquetuple(tuple):
    def __hash__(self):
        return hash(self[0])

    def __eq__(self, other):
        if not isinstance(other, uniquetuple):
            return NotImplemented
        if not self or not other:
            return len(self) == len(other)
        return self[0] == other[0]


class fileobj(NamedTuple):
    path: str
    time: datetime


def list_local(folder: str):
    folder = posixpath.join(folder, "")
    files: set[fileobj] = set()

    for root, _, _files in os.walk(folder):
        for _file in _files:
            full = posixpath.join(root, _file)
            path = full.removeprefix(folder)
            time = datetime.fromtimestamp(os.path.getmtime(full), tz=UTC)
            files.add(fileobj(path=path, time=time))

    return files


def list_remote(resource, folder: str):
    folder = posixpath.join(folder, "")
    bucket = resource.Bucket("bucket-0")  # type: ignore
    files: set[fileobj] = set()

    for obj in bucket.objects.all():
        name = obj.key
        if name.startswith(folder):
            path = name.removeprefix(folder)
            time = obj.last_modified
            files.add(fileobj(path=path, time=time))

    return files


def get_uploads(local: set[fileobj], remote: set[fileobj]):
    to_upload: set[fileobj] = set()
    remote_dict = {k: v for k, v in remote}

    for obj in local:
        l_file, l_date = obj
        if l_file not in remote_dict:
            to_upload.add(obj)
        else:
            if l_date > remote_dict[l_file]:
                to_upload.add(obj)

    return to_upload


@dataclass
class LLNCAName:
    major: str
    minor: str

    @property
    def full(self):
        return "".join([self.major, "-" if self.minor else "", self.minor])


@dataclass
class LLNCASaverConfig:
    name: LLNCAName
    s3_endpoint: str
    s3_uri: str
    folder: str
    s3_folder: str
    s3_bucket: str
    sync_interval: int
    epochs_checkpts: int
    epochs_histogram: int
    epochs_embeddings: int

    @property
    def base_folder(self):
        return posixpath.join(self.folder, self.name.full)

    @property
    def logs_folder(self):
        return posixpath.join(self.folder, self.name.full, "logs", "")

    @property
    def checkpts_folder(self):
        return posixpath.join(self.folder, self.name.full, "checkpts", "")

    def __post_init__(self):
        os.makedirs(self.base_folder, exist_ok=True)
        os.makedirs(self.logs_folder, exist_ok=True)
        os.makedirs(self.checkpts_folder, exist_ok=True)


class LLNCASaver:
    def __init__(self, config: LLNCASaverConfig):
        self.config = config
        # self.cmd = [
        #     "aws",
        #     "s3",
        #     "sync",
        #     self.config.folder,
        #     self.s3_full_uri,
        #     "--endpoint-url",
        #     self.config.s3_endpoint,
        # ]

        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self.run,
            daemon=True,
        )
        self.thread.start()

    @property
    def s3_full_uri(self):
        return posixpath.join(
            self.config.s3_uri,
            self.config.s3_folder,
            "",
        )

    def get_boto3(self):
        s3_resource = boto3.resource(
            "s3",
            endpoint_url="https://s3.krazyorange.dev",
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )
        s3_client = boto3.client(
            "s3",
            endpoint_url="https://s3.krazyorange.dev",
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )
        return s3_resource, s3_client

    def sync(self, s3_resource, s3_client):
        local = list_local(self.config.folder)
        remote = list_remote(s3_resource, self.config.s3_folder)
        uploads = get_uploads(local, remote)
        for obj in uploads:
            local_path = posixpath.join(self.config.folder, obj.path)
            remote_path = posixpath.join(self.config.s3_folder, obj.path)
            content_type, _ = mimetypes.guess_type(local_path)
            if content_type is None:
                content_type = "application/octet-stream"
            s3_client.upload_file(
                Bucket=self.config.s3_bucket,
                Filename=local_path,
                Key=remote_path,
                ExtraArgs={"ContentType": content_type},
            )

    def close(self):
        self.stop_event.set()
        self.thread.join()

        s3_resource, s3_client = self.get_boto3()
        self.sync(s3_resource, s3_client)

    def run(self):
        s3_resource, s3_client = self.get_boto3()
        while not self.stop_event.wait(timeout=self.config.sync_interval):
            try:
                self.sync(s3_resource, s3_client)
            except Exception as e:  # noqa: BLE001
                print(f"LLNCASaver S3 sync error: {e}")

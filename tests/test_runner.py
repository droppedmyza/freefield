from pathlib import Path
from types import SimpleNamespace

from comfy_sdk import ComfyError, Progress, StatusChange

from freefield import runner


class FakeOutput:
    def __init__(self, node_id: str, name: str) -> None:
        self.node_id, self.name = node_id, name

    def to_file(self, dest: Path) -> None:
        dest.write_bytes(b"clip")


class FakeJob:
    def __init__(self, status="succeeded", outputs=None, events=(), wait_error=None):
        self.id, self.status, self.error = "job-1", status, None if status == "succeeded" else "node 7 ran out of memory"
        self.outputs, self.event_list, self.wait_error = outputs or {}, list(events), wait_error

    def events(self):
        yield from self.event_list

    def wait(self, timeout):
        if self.wait_error:
            raise self.wait_error
        return self

    def get_outputs(self, node_id):
        return self.outputs.get(node_id, [])


class FakeClient:
    def __init__(self, job: FakeJob) -> None:
        self.job, self.inputs, self.closed = job, [], False
        self.workflows = SimpleNamespace(from_json=lambda graph: SimpleNamespace(
            set_input=lambda node, name, asset: self.inputs.append((node, name, asset))))
        self.assets = SimpleNamespace(from_file=lambda path: f"asset:{Path(path).name}")

    def submit(self, workflow):
        return self.job

    def close(self):
        self.closed = True


def compiled(tmp_path: Path) -> runner.CompiledRun:
    photo = tmp_path / "me.png"
    photo.write_bytes(b"png")
    return runner.CompiledRun({"photo": {"class_type": "LoadImage", "inputs": {}}}, [("photo", "image", photo)], ["clip_save"])


def test_a_run_uploads_the_inputs_streams_progress_and_downloads_the_outputs(tmp_path):
    job = FakeJob(outputs={"clip_save": [FakeOutput("clip_save", "clip_00001_.mp4")]},
                  events=[StatusChange("running"), Progress(0.5, nodes_done=3, nodes_total=6, current_node="video_sample")])
    client, seen = FakeClient(job), []
    result = runner.run(compiled(tmp_path), tmp_path / "out", client=client, on_progress=seen.append)
    assert result.ok and result.job_id == "job-1"
    assert result.files == {"clip_save": [tmp_path / "out" / "clip_save_0_clip_00001_.mp4"]}
    assert (tmp_path / "out" / "clip_save_0_clip_00001_.mp4").read_bytes() == b"clip"
    assert client.inputs == [("photo", "image", "asset:me.png")]
    assert [e["type"] for e in seen] == ["status", "status", "progress"]
    assert seen[-1]["node"] == "video_sample" and seen[-1]["nodes_done"] == 3
    assert not client.closed


def test_a_failed_job_reports_its_error_and_no_files(tmp_path):
    result = runner.run(compiled(tmp_path), tmp_path / "out", client=FakeClient(FakeJob(status="failed")))
    assert not result.ok and result.status == "failed" and result.error == "node 7 ran out of memory"
    assert result.files == {}


def test_a_job_without_output_files_is_not_ok(tmp_path):
    result = runner.run(compiled(tmp_path), tmp_path / "out", client=FakeClient(FakeJob()))
    assert not result.ok and "produced no files" in result.error


def test_a_timeout_and_an_unreachable_comfyui_are_results_not_exceptions(tmp_path):
    timed_out = runner.run(compiled(tmp_path), tmp_path / "a", client=FakeClient(FakeJob(wait_error=TimeoutError("slow"))))
    assert timed_out.status == "timeout" and "TimeoutError" in timed_out.error

    class Down(FakeClient):
        def submit(self, workflow):
            raise ComfyError("Could not reach ComfyUI.")

    down = runner.run(compiled(tmp_path), tmp_path / "b", client=Down(FakeJob()))
    assert down.status == "failed" and "Could not reach ComfyUI" in down.error

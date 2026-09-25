"""A single-worker job queue. Heavy work (training, SVDs, model downloads) runs
one job at a time so the machine stays responsive."""
import queue
import threading
import time
import traceback
import uuid

from .jsonsafe import clean


class Job:
    def __init__(self, kind, title, fn):
        self.id = uuid.uuid4().hex[:10]
        self.kind, self.title, self.fn = kind, title, fn
        self.state = "queued"
        self.progress = 0.0
        self.message = "Waiting for the previous job to finish."
        self.live = None
        self.result = None
        self.error = None
        self.created = time.time()
        self.finished = None
        self._stop = threading.Event()

    def update(self, frac, msg, live=None):
        self.progress = max(0.0, min(1.0, float(frac)))
        self.message = msg
        if live is not None:
            self.live = clean(live)

    def public(self, with_result=True):
        d = {"id": self.id, "kind": self.kind, "title": self.title, "state": self.state, "progress": round(self.progress, 4),
             "message": self.message, "error": self.error, "created": self.created, "finished": self.finished}
        if self.live is not None:
            d["live"] = self.live
        if with_result and self.state == "done":
            d["result"] = self.result
        return d


class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.q = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, kind, title, fn):
        job = Job(kind, title, fn)
        self.jobs[job.id] = job
        self.q.put(job)
        return job

    def stop(self, job_id):
        j = self.jobs.get(job_id)
        if j:
            j._stop.set()
            if j.state == "queued":
                j.state, j.error = "stopped", "Stopped before it started."
        return j

    def _worker(self):
        while True:
            job = self.q.get()
            if job.state == "stopped":
                continue
            job.state = "running"
            job.message = "Starting."
            try:
                job.result = clean(job.fn(job.update, job._stop.is_set))
                job.state = "done"
                job.progress = 1.0
            except InterruptedError as e:
                job.state, job.error = "stopped", str(e) or "Stopped."
            except Exception as e:  # surfaced to the UI verbatim
                job.state = "error"
                job.error = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            job.finished = time.time()

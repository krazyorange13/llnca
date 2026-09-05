import multiprocessing
import threading
import time
import uuid
import queue

import dearpygui.dearpygui as dpg
import numpy as np
import torch

from corpus import LLNCACorpus, LLNCACorpusConfig
from dataset import (
    LLNCADataSampler,
    LLNCADataSamplerConfig,
    LLNCADataset,
    LLNCADatasetConfig,
)
from embedding import LLNCAEmbeddings, LLNCAEmbeddingsConfig
from main import (
    LLNCA,
    LLNCAAdvConfig,
    LLNCACheckpointingConfig,
    LLNCAConfig,
    LLNCAGenConfig,
    LLNCAOptimConfig,
)
from nca import LLNCANCA, LLNCANCAConfig
from tokenizer import LLNCATokenizer, LLNCAVocab


class LLNCAProcessManager:
    POLL_RATE = 0.25

    def __init__(self):
        self.processes: list[LLNCAProcess] = []
        self.processes_lock = threading.Lock()
        self.poll_thread = threading.Thread(target=self.poll, daemon=True)
        self.poll_thread.start()

    def add_llnca(self, model_path: str):
        process = LLNCAProcess(model_path)
        process.start()
        with self.processes_lock:
            self.processes.append(process)
        return process.id

    def remove_llnca(self, id: uuid.UUID):
        with self.processes_lock:
            for process in self.processes:
                if process.id == id:
                    process.stop()

    def remove_all(self):
        for process in self.processes:
            process.stop()

    def add_input(self, user_input: str):
        for process in self.processes:
            try:
                process.inputs.put(user_input, timeout=0.1)
            except (queue.Full, queue.ShutDown):
                continue

    def poll(self):
        prev_user_input = ""
        prev_process_ids = []
        while True:
            time.sleep(LLNCAProcessManager.POLL_RATE)

            user_input = dpg.get_value("user_input_field")
            if user_input != prev_user_input:
                self.add_input(user_input)
            prev_user_input = user_input

            with self.processes_lock:
                if not dpg.is_item_hovered("llnca_process_list"):
                    self.processes = [
                        process
                        for process in self.processes
                        if process.process.is_alive()
                    ]
                curr_process_ids = [
                    (process.id, process.get_state_desc()) for process in self.processes
                ]
                if curr_process_ids != prev_process_ids:
                    self.update_process_list()
                prev_process_ids = curr_process_ids[:]

    def update_process_list(self):
        if self.processes:
            dpg.show_item("llnca_process_window")
        else:
            dpg.hide_item("llnca_process_window")

        dpg.delete_item("llnca_process_list", children_only=True)
        for process in self.processes:
            with dpg.group(horizontal=True, parent="llnca_process_list"):  # type: ignore
                dpg.add_text(process.model_path)
                dpg.add_spacer()

                button_tag = str(uuid.uuid4())

                def make_stop(button_tag, process):
                    def stop():
                        dpg.configure_item(button_tag, label="stopping", enabled=False)
                        process.stop()

                    return stop

                desc = process.get_state_desc()
                dpg.add_button(
                    tag=button_tag,
                    label=desc if desc != "alive" else "stop",
                    enabled=desc == "alive",
                    callback=make_stop(button_tag, process),
                )


class LLNCAProcess:
    POLL_RATE = 0.1
    FRAME_RATE = 0.1

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.id = str(uuid.uuid4())
        self.inputs = multiprocessing.Queue()
        self.frames = multiprocessing.Queue()
        self.process_stop_event = multiprocessing.Event()
        self.process = multiprocessing.Process(
            target=self.process_run,
            args=(self.model_path, self.process_stop_event, self.inputs, self.frames),
            daemon=True,
        )
        self.thread_stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self.thread_run,
            args=(self.thread_stop_event, self.frames),
            daemon=True,
        )

    def start(self):
        print(f"\033[2mstarting {self.model_path}\033[0m")
        self.process_stop_event.clear()
        self.process.start()

        self.window_id = self.id + "_window"
        with dpg.window(tag=self.window_id, label=self.model_path):  # type: ignore
            dpg.add_text("", tag=self.id + "_x_str")
            dpg.add_text("", tag=self.id + "_y_str")
            dpg.add_text("", tag=self.id + "_frame")

        self.thread.start()

    def stop(self):
        print(f"\033[2mstopping {self.model_path}\033[0m")
        self.process_stop_event.set()
        self.thread_stop_event.set()

        self.process.join(timeout=0.5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()

        self.inputs.close()
        self.frames.close()
        self.inputs.join_thread()
        self.frames.cancel_join_thread()

        dpg.delete_item(self.id + "_window")

    def thread_run(self, stop_event, frames: multiprocessing.Queue):
        prev_frame = ""
        while not stop_event.is_set():
            try:
                frame = frames.get(timeout=LLNCAProcess.POLL_RATE)
            except queue.Empty:
                continue
            dpg.set_value(self.id + "_x_str", frame["x_str"])
            dpg.set_value(self.id + "_y_str", frame["y_str"])
            dpg.set_value(self.id + "_frame", frame["frame"])
            if frame["frame"] != prev_frame:
                time.sleep(LLNCAProcess.FRAME_RATE)
            prev_frame = frame["frame"]

    @staticmethod
    def process_run(
        model_path: str,
        stop_event,
        inputs: multiprocessing.Queue,
        frames: multiprocessing.Queue,
    ):
        checkpoint = torch.load(
            model_path, weights_only=False, map_location=torch.device("cpu")
        )
        llnca = LLNCA(checkpoint=checkpoint)

        x_str = ""
        xs = None
        while not stop_event.is_set():
            try:
                x_str = inputs.get(timeout=LLNCAProcess.POLL_RATE)
                xs = llnca.eval_start([x_str])
            except queue.ShutDown:
                break
            except queue.Empty:
                pass

            if not x_str or xs is None:
                continue

            xs, y_pred = llnca.eval_step(xs)
            frame = llnca.tokenizer.decode(y_pred[0])

            try:
                frames.put(
                    {"x_str": x_str, "y_str": x_str, "frame": frame},
                    timeout=LLNCAProcess.POLL_RATE,
                )
            except queue.ShutDown:
                break

    def get_state_desc(self):
        stopping = self.process_stop_event.is_set()
        stopped = not self.process.is_alive()
        return "stopped" if stopped else "stopping" if stopping else "alive"


def model_path_callback(sender, app_data):
    print(sender, app_data)
    for model_path in app_data["selections"].values():
        llnca_process_manager.add_llnca(model_path)


def exit_callback():
    llnca_process_manager.remove_all()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    llnca_process_manager = LLNCAProcessManager()

    dpg.create_context()

    dpg.set_exit_callback(exit_callback)

    with dpg.font_registry():  # type: ignore
        myfont = dpg.add_font(
            "/usr/share/fonts/truetype/JetBrainsMono/JetBrainsMonoNerdFont-Medium.ttf",
            16,
        )
    dpg.bind_font(myfont)

    with dpg.file_dialog(
        tag="model_path_picker",
        show=False,
        callback=model_path_callback,
        width=600,
        height=400,
    ):  # type: ignore
        dpg.add_file_extension(".pth")
        dpg.add_file_extension(".pt")
        dpg.add_file_extension(".tar")
        dpg.add_file_extension(".*")

    with dpg.window(
        tag="main_window",
        label="llnca eval suite",
        width=300,
        height=200,
        pos=[10, 10],
    ):  # type: ignore
        dpg.add_button(
            label="load model", callback=lambda: dpg.show_item("model_path_picker")
        )

    with dpg.window(
        tag="user_input",
        label="input",
        width=300,
        height=200,
        pos=[10, 220],
    ):  # type: ignore
        dpg.add_input_text(
            tag="user_input_field",
            hint="type smth...",
        )

    with dpg.window(  # noqa: SIM117
        tag="llnca_process_window",
        label="llnca processes",
        show=False,
    ):  # type: ignore
        with dpg.group(tag="llnca_process_list"):  # type: ignore
            pass

    dpg.setup_dearpygui()

    dpg.create_viewport()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

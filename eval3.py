import functools
import multiprocessing
import pprint
import queue
import threading
import time
import uuid
import weakref

import dearpygui.dearpygui as dpg


class Unique:
    def __init__(self):
        self.id = str(uuid.uuid4())

    def __matmul__(self, other):
        if not isinstance(other, str):
            return NotImplemented

        return self.id + "-" + other


class Logger(Unique):
    def __init__(self):
        super().__init__()

    def log(self, msg: str):
        print(f"[{self.id}] {msg}")

    def logq(self, msg: str):
        print(f"\033[2m[{self.id}] {msg}\033[0m")


class LoggerException(Exception):
    pass


def auto_log(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if isinstance(self, Logger):
            self.logq(f"{self.__class__.__name__.lower()} {func.__qualname__.lower()}")
        else:
            raise LoggerException(
                f"@auto_log wraps {self.__class__.__name__}.{func.__name__} "
                f"but {self.__class__.__name__} doesn't inherit from {Logger.__name__}"
            )
        return func(self, *args, **kwargs)

    return wrapper


class Registry(Unique):
    _instances: dict | weakref.WeakValueDictionary


class WeakRegistry(Registry):
    _instances = weakref.WeakValueDictionary()

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls._instances = weakref.WeakValueDictionary()

    def __init__(self):
        super().__init__()
        self.__class__._instances[self.id] = self


class StrongRegistry(Registry):
    _instances = {}  # noqa: RUF012

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls._instances = {}

    def __init__(self):
        super().__init__()
        self.__class__._instances[self.id] = self


class Visual(StrongRegistry, Logger):
    def __init__(self):
        super().__init__()
        self.ui_thread = ThreadedPoll(
            poll_rate=1.0,
            poll_callback=self.poll_ui_thread,
            init_callback=self.open_ui,
            deinit_callback=self.close_ui,
        )
        self.ui_thread.start()

    @auto_log
    def open_ui(self):
        _instances = self.__class__._instances
        close_ref = weakref.WeakMethod(self.close_ui)

        def close_callback():
            callback = close_ref()
            if callback is not None:
                result = callback()
                return result
            return None

        with dpg.window(
            tag=self @ "window",
            label=self @ "window",
            width=300,
            height=200,
            pos=[20 * len(_instances), 20 * len(_instances)],
            autosize=True,
            on_close=lambda: close_callback(),
        ):  # type: ignore
            pass

    def poll_ui_thread(self, *args, **kwargs): ...

    def close_ui(self): ...


class VisualUnbound(Visual):
    @auto_log
    def close_ui(self):
        if dpg.does_item_exist(self @ "window"):
            dpg.configure_item(self @ "window", on_close=None)
            dpg.delete_item(self @ "window")
            self.ui_thread.stop()


class VisualBound(Visual):
    @auto_log
    def close_ui(self):
        if dpg.does_item_exist(self @ "window"):
            dpg.configure_item(self @ "window", on_close=None)
            dpg.delete_item(self @ "window")
            self.ui_thread.stop()
            self.ui_thread.poll_callback = None  # type: ignore
            self.ui_thread.init_callback = None
            self.ui_thread.deinit_callback = None
        if self.id in self.__class__._instances:
            del self.__class__._instances[self.id]


class Threaded(WeakRegistry, Logger):
    def __init__(self, use_input: bool = False, use_output: bool = False):
        super().__init__()
        self.stop_event = threading.Event()
        self.input = queue.Queue() if use_input else None
        self.output = queue.Queue() if use_output else None
        self.thread = None

    @auto_log
    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self.run,
            args=(
                *([self.input] if self.input is not None else []),
                *([self.output] if self.output is not None else []),
            ),
            daemon=True,
        )
        self.thread.start()

    @auto_log
    def stop(self):
        self.stop_event.set()
        if self.input is not None:
            self.input.shutdown()
        if self.output is not None:
            self.output.shutdown()
        curr_thread = threading.current_thread()
        if self.thread is not None and curr_thread != self.thread:
            self.thread.join()

    @auto_log
    def run(self, *args, **kwargs): ...


class ThreadedPoll(Threaded):
    def __init__(
        self,
        poll_rate=1.0,
        poll_callback=lambda *args, **kwargs: None,
        init_callback=lambda *args, **kwargs: None,
        deinit_callback=lambda *args, **kwargs: None,
        use_input=False,
        use_output=False,
    ):
        super().__init__(use_input=use_input, use_output=use_output)
        self.poll_rate = poll_rate
        self.poll_callback = poll_callback
        self.init_callback = init_callback
        self.deinit_callback = deinit_callback

    @auto_log
    def run(self, *args, **kwargs):
        self.init_callback(*args, **kwargs)
        self.poll_callback(*args, **kwargs)
        while not self.stop_event.wait(timeout=self.poll_rate):
            self.poll_callback(*args, **kwargs)
        self.deinit_callback(*args, **kwargs)


class Multiprocessed(WeakRegistry, Logger):
    def __init__(self, use_input=False, use_output=False):
        super().__init__()
        self.stop_event = multiprocessing.Event()
        self.input = multiprocessing.Queue() if use_input else None
        self.output = multiprocessing.Queue() if use_output else None
        self.process = None

    @auto_log
    def start(self):
        self.stop_event.clear()
        self.process = multiprocessing.Process(
            target=self.run,
            args=(
                *([self.input] if self.input is not None else []),
                *([self.output] if self.output is not None else []),
            ),
            daemon=True,
        )
        self.process.start()

    @auto_log
    def stop(self):
        if self.process is None:
            return

        self.stop_event.set()
        self.process.join(timeout=0.5)

        if self.process.is_alive():
            self.process.terminate()
            self.process.join()

        if self.input is not None:
            self.input.close()
            self.input.join_thread()
        if self.output is not None:
            self.output.close()
            self.output.join_thread()

    @auto_log
    def run(
        self,
        input: multiprocessing.Queue,
        output: multiprocessing.Queue,
    ): ...


class Inspector(VisualBound):
    def __init__(self, obj):
        self.ref = weakref.ref(obj)
        super().__init__()

    def poll_ui_thread(self, *args, **kwargs):
        if not dpg.does_item_exist(self @ "window"):
            self.ui_thread.stop_event.set()
            return
        if self.ref() is None:
            self.ui_thread.stop_event.set()
            return
        dpg.delete_item(self @ "window", children_only=True)
        dpg.add_text(f"inspecting {self.ref()}", parent=self @ "window")
        dpg.add_text(pprint.pformat(vars(self.ref()), indent=4), parent=self @ "window")


class RegistryManager(VisualBound):
    def __init__(self, *registries: type[Registry]):
        self.registries = registries
        super().__init__()

    def poll_ui_thread(self, *args, **kwargs):
        if not dpg.does_item_exist(self @ "window"):
            self.ui_thread.stop_event.set()
            return
        dpg.delete_item(self @ "window", children_only=True)
        dpg.add_text(
            f"registry manager for {' '.join([registry.__name__ for registry in self.registries])}",
            parent=self @ "window",
        )
        for registry in self.registries:
            for instance in list(registry._instances.values()):
                with dpg.group(horizontal=True, parent=self @ "window"):  # type: ignore

                    def launch_inspect(instance):
                        def wrapped():
                            Inspector(instance)

                        return wrapped

                    dpg.add_text(instance.id)
                    dpg.add_button(label="inspect", callback=launch_inspect(instance))


class ProcessManager(VisualBound):
    pass


class Process(VisualUnbound):
    pass


def exit_callback():
    pass


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    dpg.create_context()
    dpg.set_exit_callback(exit_callback)

    with dpg.window(label="main"):  # type: ignore
        dpg.add_button(
            label="enumerate threads",
            callback=lambda: print(threading.enumerate()),
        )

    RegistryManager(Threaded, ThreadedPoll)
    RegistryManager(Inspector)

    dpg.setup_dearpygui()
    dpg.create_viewport()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

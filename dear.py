import random
import threading

import dearpygui.dearpygui as dpg
import numpy as np

rng = np.random.default_rng()

img = np.zeros((256, 256, 3), dtype=np.float32)
img_data = img.ravel()


def data_thread():
    chars = [*range(32, 127), *range(256, 292)]
    while True:
        dpg.set_value(
            "stream", f"{''.join([chr(random.choice(chars)) for _ in range(16)])}"
        )
        rng.random(out=img_data, dtype=np.float32)


dpg.create_context()

with dpg.font_registry():  # type: ignore
    myfont = dpg.add_font(
        "/usr/share/fonts/truetype/JetBrainsMono/JetBrainsMonoNerdFont-Medium.ttf", 18
    )
dpg.bind_font(myfont)


with dpg.texture_registry(show=False):  # type: ignore
    dpg.add_raw_texture(
        width=256,
        height=256,
        default_value=img_data,  # type: ignore
        format=dpg.mvFormat_Float_rgb,
        tag="texture_tag",
    )

with dpg.window(label="hi wrld", width=600, height=400):  # type: ignore
    dpg.add_text("", tag="stream")
    dpg.add_image(texture_tag="texture_tag", width=1024, height=1024)

dpg.create_viewport(title="hai world", width=600, height=400, vsync=False)
dpg.show_metrics()
dpg.setup_dearpygui()
dpg.show_viewport()

thread = threading.Thread(target=data_thread, args=(), daemon=True)
thread.start()

dpg.start_dearpygui()

dpg.destroy_context()

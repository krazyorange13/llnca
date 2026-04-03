import os
import sys
import subprocess

os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

import cv2
import numpy as np
import torch
from tqdm import tqdm

from main import LLNCA, LLNCAConfig, Pool


class Visualization:
    def __init__(
        self,
        path: str,
        frame_count: int,
        mod: str | None = None,
        ffmpeg: bool | None = None,
    ):
        self.path = path
        state = torch.load(self.path, weights_only=False)
        self.llnca = LLNCA(state["config"], state)
        self.frames = []
        self.frame_count = frame_count

        self.mod = mod
        self.ffmpeg = ffmpeg

        if self.mod:
            self.name = f"{self.llnca.config.name}-{self.mod}"
        else:
            self.name = self.llnca.config.name

    def viz(self):
        self.generate_frames()
        self.save_frames()
        if self.ffmpeg:
            self.generate_movie()
        self.display_frames()

    def generate_frames(self):
        print("rendering animation...")
        pool = Pool(
            self.llnca.dataset,
            bin=1,
            renderer=self.llnca.renderer,
            channels=self.llnca.config.channels,
        )
        y, x, f = pool.get_row()
        x = x.unsqueeze(0)
        f = f.unsqueeze(0)
        for i in tqdm(range(self.frame_count), leave=False):
            with torch.no_grad():
                x = self.llnca.nca.step(x, f)
            self.frames.append(self.nca_to_img(x))

    def nca_to_img(self, x: torch.Tensor):
        y = torch.clamp(x[0, :1], 0.0, 1.0)
        y = 1 - y
        # rgb = y[:3]
        # alpha = y[3:4]
        # bg_color = 255.0
        # blended = rgb * alpha + bg_color * (1.0 - alpha)
        img = y.detach().permute(1, 2, 0).numpy()
        return img

    def save_frames(self):
        print("saving animation...")
        for i in tqdm(range(len(self.frames)), leave=False):
            frame = self.frames[i]
            j = str(i).zfill(len(str(len(self.frames) - 1)))
            cv2.imwrite(
                f"anim/{self.name}-{j}.png",
                (frame * 255.0).astype(np.uint8),
            )

    def generate_movie(self):
        print("generating movie... ", end="", flush=True)
        d = str(len(str(len(self.frames) - 1))).zfill(2)
        anim_in = f"anim/{self.name}-%{d}d.png"
        anim_out = f"movies/{self.name}.mp4"
        scale_factor = 16
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                "60",
                "-i",
                anim_in,
                "-vf",
                f"scale={scale_factor}*iw:-1:flags=neighbor,format=yuv444p",
                "-c:v",
                "libx264",
                "-crf",
                "5",
                "-preset",
                "veryslow",
                "-tune",
                "animation",
                anim_out,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"saved to {anim_out}")

    def display_frames(self):
        print("displaying animation...")
        quit = False
        window = self.name
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while not quit:
            for i in tqdm(range(len(self.frames)), leave=False):
                frame = self.frames[i]
                cv2.imshow(window, frame)
                if cv2.waitKey(int(1000 / 60)) & 0xFF == ord("q"):
                    quit = True
                    break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) not in [2, 3]:
        print("usage uwu: python viz.py model [modifier]")
        exit(1)

    args = sys.argv[:]

    ffmpeg = "-ffmpeg" in args
    if ffmpeg:
        args.remove("-ffmpeg")

    path = args[1]
    mod = args[2] if 2 < len(args) else None

    viz = Visualization(path, 1000, mod=mod, ffmpeg=ffmpeg)
    viz.viz()

"""V2 视频分段写出与断点续传（`src/v2/video_segments.py`）。"""

import os
import shutil
import tempfile
import unittest

import imageio.v3 as iio
import numpy as np

from src.v2 import video_segments as VS

W, H, FPS = 64, 48, 24


def _frame(f):
    """第 f 帧的确定性测试图案（亮度随帧号变化，便于核对帧序）。"""
    img = np.zeros((H, W, 3), np.uint8)
    img[:, :, 0] = (f * 7) % 256
    img[:, : W // 2, 1] = 200
    return img


class SegmentedRenderTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.out = os.path.join(self.dir, "v.mp4")
        self.params = {"n_frames": 25, "fov": 34.0}
        self.calls = []

    def tearDown(self):
        shutil.rmtree(self.dir)

    def _frames(self, stop_at=None):
        """返回 render_frames 回调；stop_at 不为 None 时在该帧抛异常模拟中断。"""
        def gen(f0, f1, exposure):
            # 与 CLI 一致：曝光为 None 时由本段首帧计算（固定 0.5），之后各帧沿用
            for f in range(f0, f1):
                if f == stop_at:
                    raise KeyboardInterrupt
                self.calls.append((f, exposure))
                if exposure is None:
                    exposure = 0.5
                yield _frame(f), exposure
        return gen

    def _check_video(self, n):
        frames = list(iio.imiter(self.out, plugin="pyav"))
        self.assertEqual(len(frames), n)
        for f in (0, 9, 10, n - 1):
            self.assertLess(abs(float(frames[f][:, :, 0].mean()) - (f * 7) % 256), 4.0)

    def test_full_run_concats_and_cleans_up(self):
        VS.render_segmented(25, FPS, self.out, self.params, False, self._frames(), seg_frames=10, log=lambda m: None)
        self._check_video(25)
        self.assertFalse(os.path.isdir(VS.segment_dir(self.out)))
        # 曝光只在第 0 帧计算一次，之后所有帧都拿到锁定值
        self.assertIsNone(self.calls[0][1])
        self.assertTrue(all(e == 0.5 for _f, e in self.calls[1:]))

    def test_resume_skips_finished_segments(self):
        with self.assertRaises(KeyboardInterrupt):
            VS.render_segmented(25, FPS, self.out, self.params, False, self._frames(stop_at=15),
                                seg_frames=10, log=lambda m: None)
        seg = VS.segment_dir(self.out)
        self.assertTrue(os.path.isfile(VS.segment_path(seg, 0)))
        self.assertFalse(os.path.isfile(VS.segment_path(seg, 1)))  # 未写完的分段不留正式文件
        self.calls.clear()
        VS.render_segmented(25, FPS, self.out, self.params, True, self._frames(), seg_frames=10, log=lambda m: None)
        # 只重渲第 1 段起的帧，且使用上次锁定的曝光
        self.assertEqual(self.calls[0], (10, 0.5))
        self.assertEqual(len(self.calls), 15)
        self._check_video(25)

    def test_changed_params_start_over(self):
        with self.assertRaises(KeyboardInterrupt):
            VS.render_segmented(25, FPS, self.out, self.params, False, self._frames(stop_at=15),
                                seg_frames=10, log=lambda m: None)
        self.calls.clear()
        VS.render_segmented(25, FPS, self.out, {**self.params, "fov": 40.0}, True, self._frames(),
                            seg_frames=10, log=lambda m: None)
        self.assertEqual(self.calls[0], (0, None))
        self._check_video(25)


if __name__ == "__main__":
    unittest.main()

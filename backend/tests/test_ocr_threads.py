"""本地 OCR 的 CPU 线程上限（降低屏幕共享/录制等高负载场景的 CPU 争用）。

背景：rapidocr_onnxruntime 的 OrtInferSession 构造 SessionOptions() 后从不设置
intra_op_num_threads，onnxruntime 便按物理核数起线程、det/cls/rec 三会话各自都想
吃满全核。屏幕共享时 OCR 一起来会把 CPU 占满，饿死代理转发与浏览器渲染。
我们在构造 RapidOCR 前把 SessionOptions 换成预设线程上限的工厂来注入限制。
"""
from __future__ import annotations

import os

import pytest

from app.config import settings
from app.providers import ocr


@pytest.fixture
def rocr_utils():
    """拿到 rapidocr 的 utils 模块，并在用例后还原 SessionOptions，避免补丁外溢。

    未安装 rapidocr 时整组「打补丁」相关用例自动跳过（它是可选依赖）。
    """
    mod = pytest.importorskip("rapidocr_onnxruntime.utils")
    original = mod.SessionOptions
    try:
        yield mod
    finally:
        mod.SessionOptions = original


# --------------------------------------------------------------------------
# 解析器：配置值 → 实际线程数
# --------------------------------------------------------------------------
def test_resolver_maps_config(monkeypatch):
    """<0=不干预(0)，>0=原值，0=自动 max(2, cpu//2)。"""
    monkeypatch.setattr(settings, "ocr_cpu_threads", -1)
    assert ocr.effective_ocr_threads() == 0

    monkeypatch.setattr(settings, "ocr_cpu_threads", 3)
    assert ocr.effective_ocr_threads() == 3

    monkeypatch.setattr(settings, "ocr_cpu_threads", 0)
    cores = os.cpu_count() or 4
    assert ocr.effective_ocr_threads() == max(2, cores // 2)


def test_resolver_auto_floor_is_two_on_tiny_cpu(monkeypatch):
    """自动档在单/双核机上也不会退到 1，保底 2 交给 onnxruntime 自行调度。"""
    monkeypatch.setattr(settings, "ocr_cpu_threads", 0)
    monkeypatch.setattr(ocr.os, "cpu_count", lambda: 1)
    assert ocr.effective_ocr_threads() == 2


# --------------------------------------------------------------------------
# 打补丁：SessionOptions 工厂是否真的带上 intra_op_num_threads
# --------------------------------------------------------------------------
def test_limit_sets_intra_op_threads(rocr_utils):
    ocr.limit_rapidocr_threads(3)
    so = rocr_utils.SessionOptions()
    assert so.intra_op_num_threads == 3


def test_limit_noop_when_non_positive(rocr_utils):
    """n<=0 时必须完全不干预（回滚路径：配置 <0 = 引擎默认）。"""
    before = rocr_utils.SessionOptions
    ocr.limit_rapidocr_threads(0)
    assert rocr_utils.SessionOptions is before
    assert not getattr(rocr_utils.SessionOptions, "_campus_thread_capped", False)


def test_limit_is_idempotent(rocr_utils):
    """重复调用不得层层包裹：第二次被幂等守卫忽略，仍用第一次的上限。"""
    ocr.limit_rapidocr_threads(3)
    first = rocr_utils.SessionOptions
    ocr.limit_rapidocr_threads(5)
    assert rocr_utils.SessionOptions is first
    assert rocr_utils.SessionOptions().intra_op_num_threads == 3
